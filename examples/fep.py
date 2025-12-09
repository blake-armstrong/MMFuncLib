"""
OpenMM Free Energy Perturbation (FEP) Module
Clean, object-oriented FEP implementation for OpenMM simulations
"""

import sys
import os
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import logging
logging.getLogger().setLevel(logging.DEBUG)

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from mmfunclib import FEPReporter, FEPConfig

@dataclass
class FEPParams:
    """Container for Free Energy Perturbation parameters"""
    m: float = 2.0
    n: float = 1.0
    alpha: float = 0.5
    atoms_to_fep: List[int] = None
    atoms_to_restrain: List[int] = None
    restraint_k: float = 14473.0  # kJ/mol/nm^2
    
    def __post_init__(self):
        if self.atoms_to_fep is None:
            self.atoms_to_fep = []
        if self.atoms_to_restrain is None:
            self.atoms_to_restrain = []


class FEPSystem:
    """Handles FEP-specific modifications to OpenMM systems"""
    
    def __init__(self, fep_params: FEPParams):
        self.params = fep_params
    
    def apply_lambda_scaling(self, system, slmb: float, elmb: float):
        """Apply lambda scaling to nonbonded interactions"""
        self._scale_electrostatics(system, elmb)
        self._scale_steric_interactions(system, slmb)
    
    def _scale_electrostatics(self, system, elmb: float):
        """Scale electrostatic interactions by lambda"""
        nonbonded = next(f for f in system.getForces() 
                        if isinstance(f, mm.NonbondedForce))
        for atom_id in self.params.atoms_to_fep:
            charge, sigma, epsilon = nonbonded.getParticleParameters(atom_id)
            scaled_charge = charge * unit.Quantity(value=elmb)
            nonbonded.setParticleParameters(atom_id, scaled_charge, sigma, epsilon)
    
    def _scale_steric_interactions(self, system, slmb: float):
        """Scale steric interactions using soft-core potential"""
        custom_nonbonded = [f for f in system.getForces() 
                           if isinstance(f, mm.CustomNonbondedForce)]
        
        for force in custom_nonbonded:
            # Zero out parameters for FEP atoms
            for atom_id in self.params.atoms_to_fep:
                params = list(force.getParticleParameters(atom_id))
                params[1] = 0
                force.setParticleParameters(atom_id, params)
            
            # Update soft-core parameters
            energy_func = force.getEnergyFunction()
            energy_func = (energy_func
                          .replace("l=1", f"l={slmb}")
                          .replace("m=2", f"m={self.params.m}")
                          .replace("n=1", f"n={self.params.n}")
                          .replace("a=0.5", f"a={self.params.alpha}"))
            force.setEnergyFunction(energy_func)
    
    def add_restraints(self, system, positions, rlmb: float = 1.0):
        """Add harmonic restraints to specified atoms"""
        if not self.params.atoms_to_restrain:
            return
        
        restraint = mm.CustomExternalForce(
            "0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2"
        )
        restraint.addGlobalParameter("k", 
            self.params.restraint_k * kilojoules_per_mole / nanometer**2)
        restraint.addGlobalParameter("lmb", rlmb)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        
        for atom_id in self.params.atoms_to_restrain:
            restraint.addParticle(atom_id, positions[atom_id])
        
        system.addForce(restraint)


class LambdaSchedule:
    """Manages lambda value schedules for FEP"""
    
    @staticmethod
    def get_default_schedules() -> Tuple[np.ndarray, np.ndarray]:
        """Get default electrostatic and steric lambda schedules"""
        # Electrostatic lambdas (turn off charges first)
        e_lambdas = np.array([
            1.0, 0.96551724, 0.93103448, 0.89655172, 0.86206897, 0.82758621,
            0.79310345, 0.75862069, 0.72413793, 0.68965517, 0.65517241, 0.62068966,
            0.5862069, 0.55172414, 0.51724138, 0.48275862, 0.44827586, 0.4137931,
            0.37931034, 0.34482759, 0.31034483, 0.27586207, 0.24137931, 0.20689655,
            0.17241379, 0.13793103, 0.10344828, 0.06896552, 0.03448276, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        
        # Steric lambdas (turn off VDW after charges)
        s_lambdas = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.8638376, 0.74063276, 0.62973761, 0.53050427, 0.44228485, 0.36443149,
            0.2962963, 0.2372314, 0.18658892, 0.14372098, 0.1079797, 0.0787172,
            0.05528561, 0.03703704, 0.02332362, 0.01349746, 0.0069107, 0.00291545,
            0.00086384, 0.00010798, 0.0
        ])
        
        return e_lambdas, s_lambdas
    
   # @staticmethod
   # def split_for_multi_gpu(e_lambdas: np.ndarray, s_lambdas: np.ndarray,
   #                        gpu_id: int, n_gpus: int) -> Tuple[np.ndarray, np.ndarray]:
   #     """Split lambda schedules across multiple GPUs with overlap"""
   #     e_split = np.array_split(e_lambdas, n_gpus)
   #     s_split = np.array_split(s_lambdas, n_gpus)
   #     
   #     if gpu_id == 0:
   #         # First GPU: add overlap with next
   #         e_out = np.concatenate([e_split[gpu_id], [e_split[gpu_id + 1][0]]])
   #         s_out = np.concatenate([s_split[gpu_id], [s_split[gpu_id + 1][0]]])
   #     elif gpu_id == n_gpus - 1:
   #         # Last GPU: add overlap with previous
   #         e_out = np.concatenate([[e_split[gpu_id - 1][-1]], e_split[gpu_id]])
   #         s_out = np.concatenate([[s_split[gpu_id - 1][-1]], s_split[gpu_id]])
   #     else:
   #         # Middle GPUs: add overlap with both neighbors
   #         e_out = np.concatenate([
   #             [e_split[gpu_id - 1][-1]], 
   #             e_split[gpu_id], 
   #             [e_split[gpu_id + 1][0]]
   #         ])
   #         s_out = np.concatenate([
   #             [s_split[gpu_id - 1][-1]], 
   #             s_split[gpu_id], 
   #             [s_split[gpu_id + 1][0]]
   #         ])
   #     
   #     return e_out, s_out

    @staticmethod
    def split_for_multi_gpu(e_lambdas: np.ndarray, s_lambdas: np.ndarray,
                           gpu_id: int, n_gpus: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Split lambda schedules across multiple GPUs with overlap

        Returns:
            e_out: Electrostatic lambdas for this GPU
            s_out: Steric lambdas for this GPU
            offset: Starting index in the original lambda array
        """
        e_split = np.array_split(e_lambdas, n_gpus)
        s_split = np.array_split(s_lambdas, n_gpus)

        # Calculate offset (starting index in original array)
        offset = sum(len(e_split[i]) for i in range(gpu_id))

        if gpu_id == 0:
            # First GPU: add overlap with next
            e_out = np.concatenate([e_split[gpu_id], [e_split[gpu_id + 1][0]]])
            s_out = np.concatenate([s_split[gpu_id], [s_split[gpu_id + 1][0]]])
        elif gpu_id == n_gpus - 1:
            # Last GPU: add overlap with previous
            e_out = np.concatenate([[e_split[gpu_id - 1][-1]], e_split[gpu_id]])
            s_out = np.concatenate([[s_split[gpu_id - 1][-1]], s_split[gpu_id]])
            offset -= 1  # Account for overlap window from previous GPU
        else:
            # Middle GPUs: add overlap with both neighbors
            e_out = np.concatenate([
                [e_split[gpu_id - 1][-1]],
                e_split[gpu_id],
                [e_split[gpu_id + 1][0]]
            ])
            s_out = np.concatenate([
                [s_split[gpu_id - 1][-1]],
                s_split[gpu_id],
                [s_split[gpu_id + 1][0]]
            ])
            offset -= 1  # Account for overlap window from previous GPU

        return e_out, s_out, offset


class FEPRunner:
    """Runs FEP calculations across lambda windows"""
    
    def __init__(self, fep_params: FEPParams):
        self.fep = fep_params
        self.fep_system = FEPSystem(fep_params)
    
    def run(self, 
            system_builder_func,
            e_lambdas: np.ndarray,
            s_lambdas: np.ndarray,
            output_dir: str = "FEP",
            skip_windows: Optional[List[int]] = None,
            window_offset=0):

        """
        Run FEP calculations across all lambda windows
        
        Args:
            system_builder_func: Function that builds and returns (system, topology, positions)
            e_lambdas: Electrostatic lambda values
            s_lambdas: Steric lambda values
            output_dir: Base directory for FEP outputs
            skip_windows: List of window indices to skip (for multi-GPU)
            window_offset: Offset for window numbering (for multi-GPU)

        """
        if len(e_lambdas) != len(s_lambdas):
            raise ValueError("Lambda arrays must have same length")
        
        if skip_windows is None:
            skip_windows = []
        
        cwd = os.getcwd()
        
        print("<<-------- FEP Simulation ------->>")
        print(f"Performing FEP on atoms: {self.fep.atoms_to_fep}")
        print(f"Number of lambda windows: {len(e_lambdas)}")
        print("<<-------------------------------->>") 

        
        for window_idx, (elmb, slmb) in enumerate(zip(e_lambdas, s_lambdas)):
            if window_idx in skip_windows:
                continue
            
            # Calculate global window index
            global_idx = window_offset + window_idx

            # Setup window directory
            window_dir = f"lambda_{global_idx:03d}"
            window_path = os.path.join(cwd, output_dir, window_dir)
            os.makedirs(window_path, exist_ok=True)
            os.chdir(window_path)
            
            print(f"\n<<------- Window {window_idx} ------>>")
            print(f"Electrostatic λ = {elmb:.6f}")
            print(f"Steric λ = {slmb:.6f}")
            
            # Build systems for this window and neighbors
            system, topology, positions = system_builder_func()
            
            # Apply FEP modifications
            self.fep_system.apply_lambda_scaling(system, slmb, elmb)
            self.fep_system.add_restraints(system, positions)
            
            # Build neighbor systems if not at boundaries
            system_prev = None
            system_next = None
            
            if window_idx > 0:
                system_prev, _, _ = system_builder_func()
                self.fep_system.apply_lambda_scaling(
                    system_prev, s_lambdas[window_idx - 1], e_lambdas[window_idx - 1]
                )
            
            if window_idx < len(e_lambdas) - 1:
                system_next, _, _ = system_builder_func()
                self.fep_system.apply_lambda_scaling(
                    system_next, s_lambdas[window_idx + 1], e_lambdas[window_idx + 1]
                )
            
            # Return systems for user to create simulations
            yield {
                'window_idx': window_idx,
                'elmb': elmb,
                'slmb': slmb,
                'system': system,
                'system_prev': system_prev,
                'system_next': system_next,
                'topology': topology,
                'positions': positions,
                'path': window_path
            }
            
            os.chdir(cwd)


def calculate_window_start_index(gpu_id: int, n_gpus: int, 
                                 total_windows: int) -> int:
    """Calculate the starting window index for a given GPU"""
    windows_per_gpu = total_windows // n_gpus
    extra = total_windows % n_gpus
    
    if gpu_id < extra:
        return gpu_id * (windows_per_gpu + 1)
    else:
        return gpu_id * windows_per_gpu + extra


def get_skip_windows_for_gpu(gpu_id: int, n_gpus: int, 
                             n_windows: int) -> List[int]:
    """
    Determine which window indices to skip for multi-GPU setup
    to avoid duplicating overlap windows
    """
    skip = []
    
    # First GPU: skip last window (overlap with next GPU)
    if gpu_id == 0:
        skip.append(n_windows - 1)
    # Last GPU: skip first window (overlap with previous GPU)
    elif gpu_id == n_gpus - 1:
        skip.append(0)
    # Middle GPUs: skip first and last windows
    else:
        skip.extend([0, n_windows - 1])
    
    return skip


# ============================================================================
# MAIN SIMULATION SETUP
# ============================================================================

def main():
    """Main simulation configuration and execution"""
    
    # Initialize random seed
    random.seed(int(time.time()))
    
    # ========================================================================
    # SIMULATION PARAMETERS - CONFIGURE HERE
    # ========================================================================
    
    # Input files
    coord_file = sys.argv[1]
    forcefield_file = sys.argv[2]

    # Multi-GPU settings
    use_multi_gpu = True
    n_gpus = 8
    gpu_id = int(os.environ["SYSTEM_ID"])  # Which GPU this process uses
    
    # Platform
    platform = mm.Platform.getPlatformByName('HIP')
    properties = {'Precision': 'mixed', 'DeviceIndex': str(0)}
    
    # Thermodynamic state
    temperature = 300.0  # Kelvin
    pressure = 1.0  # atmospheres
    
    # Integration parameters
    timestep = 0.001  # picoseconds
    n_steps = 10000000
    trel = 1.0  # thermostat relaxation time (1/ps)
    
    # Output settings
    traj_frequency = 10000  # steps between trajectory frames
    thermo_frequency = 10000  # steps between thermodynamic outputs
   
    pdb = app.PDBFile(coord_file)
    residue_id_to_fep = 1
    atoms_to_fep = [int(a.index) for res in pdb.topology.residues() for a in res.atoms() if int(res.id) == residue_id_to_fep]
        
        
    # FEP settings
    fep_params = FEPParams(
        m=2.0,
        n=1.0,
        alpha=0.5,
        atoms_to_fep=atoms_to_fep,  # Atom IDs to apply FEP
        atoms_to_restrain=[],  # Atom IDs to restrain
        restraint_k=14473.0  # kJ/mol/nm^2
    )

    fep_config = FEPConfig(filename = "fep.out", report_interval = 1000)
    
    
    # ========================================================================
    # LAMBDA SCHEDULE SETUP
    # ========================================================================
    
    e_lambdas, s_lambdas = LambdaSchedule.get_default_schedules()
    # e_lambdas = [1.0, 0.5, 0.0, 0.0, 0.0]
    # s_lambdas = [1.0, 1.0, 1.0, 0.5, 0.0]
   
    window_offset = 0
    if use_multi_gpu:
        e_lambdas, s_lambdas, window_offset = LambdaSchedule.split_for_multi_gpu(
            e_lambdas, s_lambdas, gpu_id, n_gpus
        )
        skip_windows = get_skip_windows_for_gpu(gpu_id, n_gpus, len(e_lambdas))
    else:
        skip_windows = []
    
    # ========================================================================
    # SYSTEM BUILDER FUNCTION
    # ========================================================================
    
    def build_system():
        """Build OpenMM system - customize this for your system"""
        # Load structure
        pdb = app.PDBFile(coord_file)
        forcefield = app.ForceField(forcefield_file)
        
        # Create system
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            ewaldErrorTolerance=1e-05,
            nonbondedCutoff=1.0 * unit.nanometer,
            useDispersionCorrection=False,
            constraints=None,
            rigidWater=True
        )
        
        # Add thermostat
        temp = temperature * unit.kelvin
        ts = timestep * unit.picosecond
        tau = trel / unit.picosecond

        for n, f in enumerate(system.getForces()):
            f.setForceGroup(n + 1)
    
        # Create the integrator object
        integrator = mm.LangevinMiddleIntegrator(
            temp, tau, ts
        )
         
        return system, pdb.topology, pdb.positions, integrator
    
    # ========================================================================
    # RUN FEP CALCULATIONS
    # ========================================================================
    
    runner = FEPRunner(fep_params)
    
    for window_data in runner.run(
        lambda: build_system()[:3],  # Return only system, topology, positions
        e_lambdas,
        s_lambdas,
        skip_windows=skip_windows,
        window_offset=window_offset
    ):
        # Build integrator (not modified by FEP)
        _, _, _, integrator = build_system()
        
        # Create simulation
        simulation = mm.app.Simulation(
            window_data['topology'],
            window_data['system'],
            integrator,
            platform,
            properties
        )
        
        # Initialize state
        simulation.context.setPositions(window_data['positions'])
        
        simulation.context.setVelocitiesToTemperature(
            temperature * unit.kelvin, 
            random.randrange(99999)
        )
        
        # Add reporters
        simulation.reporters.append(
            app.StateDataReporter(
                'output.dat',
                thermo_frequency,
                step=True,
                time=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True
            )
        )
        
        simulation.reporters.append(
            app.DCDReporter(
                'trajectory.dcd',
                traj_frequency
            )
        )
        
        simulation.reporters.append(
            app.StateDataReporter(
                sys.stdout,
                int(n_steps / 50),
                step=True,
                potentialEnergy=True,
                temperature=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=n_steps
            )
        )
        
        # Add FEP reporter if neighbor systems exist
        if window_data['system_prev'] or window_data['system_next']:
            # Create neighbor simulations for energy evaluations
            sim_prev = None
            sim_next = None
            contexts = [] 
            if window_data['system_prev']:
                sim_prev = mm.app.Simulation(
                    window_data['topology'],
                    window_data['system_prev'],
                    mm.VerletIntegrator(1),
                    platform,
                    properties
                )
                sim_prev.context.setPositions(window_data['positions'])
                contexts.append(sim_prev.context)
            
            if window_data['system_next']:
                sim_next = mm.app.Simulation(
                    window_data['topology'],
                    window_data['system_next'],
                    mm.VerletIntegrator(1),
                    platform,
                    properties
                )
                sim_next.context.setPositions(window_data['positions'])
                contexts.append(sim_next.context)
            
            is_first = window_data['window_idx'] == 0
            is_last = window_data['window_idx'] == len(e_lambdas) - 1
            
            simulation.reporters.append(
                FEPReporter(contexts, fep_config)
            )
        
        # Run simulation
        print(f"Running {n_steps} steps...")
        simulation.step(n_steps)
        
        # Save checkpoint
        simulation.saveCheckpoint('checkpoint.chk')
        
        print(f"Window {window_data['window_idx']} complete!\n")


if __name__ == "__main__":
    main()
