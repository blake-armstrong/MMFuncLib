"""
OpenMM Free Energy Perturbation (FEP) Module
Clean, object-oriented FEP implementation for OpenMM simulations
"""

import sys
import os
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable

import numpy as np
import logging
logging.getLogger().setLevel(logging.DEBUG)

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from mmfunclib import FEPConfig, File

LOG = logging.getLogger(__name__)

@dataclass
class FEPParams:
    """Container for Free Energy Perturbation parameters"""
    elmbs: List[float]
    slmbs: List[float]
    atoms_to_fep: List[int]
    atoms_to_restrain: List[int]
    m: float = 2.0
    n: float = 1.0
    alpha: float = 0.5
    restraint_k: float = 14473.0  # kJ/mol/nm^2
    
DEFAULT_CONFIG = FEPConfig()

class FEPSystem:
    """Handles FEP-specific modifications to OpenMM systems"""
    
    def __init__(self, context, fep_params: FEPParams):
        self.params = fep_params
        self.context = context
        self._init_fep_flags()
        self._init_electrostatics()
        self.context.reinitialize(preserveState=False)

    def __call__(self, elmb, slmb):
        self.apply_lambda_scaling(elmb, slmb)

    def apply_lambda_scaling(self, elmb, slmb):
        """Apply lambda scaling to nonbonded interactions"""
        self.context.setParameter("elmb", 1 - elmb)
        self.context.setParameter("l", slmb)

    def _init_electrostatics(self):
        nonbonded = next(f for f in self.context.getSystem().getForces() 
                        if isinstance(f, mm.NonbondedForce))
        nonbonded.addGlobalParameter("elmb", 1.0)
        nonbonded.updateParametersInContext(self.context)
        for atom_id in self.params.atoms_to_fep:
            charge, _, _ = nonbonded.getParticleParameters(atom_id)
            nonbonded.addParticleParameterOffset("elmb", atom_id, -1 * charge, 0.0, 0.0)


    def _init_fep_flags(self):
        """Scale steric interactions using soft-core potential"""
        custom_nonbonded = [f for f in self.context.getSystem().getForces() 
                           if isinstance(f, mm.CustomNonbondedForce)]

        self.context.setParameter("m", self.params.m)
        self.context.setParameter("n", self.params.n)
        self.context.setParameter("a", self.params.alpha)
        
        for force in custom_nonbonded:
            # Zero out parameters for FEP atoms
            for atom_id in self.params.atoms_to_fep:
                params = list(force.getParticleParameters(atom_id))
                params[1] = 0
                force.setParticleParameters(atom_id, params)

    
    def add_restraints(self, system, positions, rlmb: float = 1.0):
        """Add harmonic restraints to specified atoms"""
        if not self.params.atoms_to_restrain:
            return
        
        restraint = mm.CustomExternalForce(
            "0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2"
        )
        restraint.addGlobalParameter("k", 
            self.params.restraint_k * unit.kilojoules_per_mole / unit.nanometer**2)
        restraint.addGlobalParameter("lmb", rlmb)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        
        for atom_id in self.params.atoms_to_restrain:
            restraint.addParticle(atom_id, positions[atom_id])
        
        system.addForce(restraint)

class FEPReporter:
    FMT = "{:20.8f}"
    HEADER = "{:>20}"

    def __init__(
        self, dummy_context: mm.Context, fep_params: FEPParams, fep_system: FEPSystem, config: FEPConfig = DEFAULT_CONFIG
    ) -> None:
        LOG.debug("Intialising FEPReporter")
        self.dummy_context = dummy_context
        self.fep_params = fep_params
        self.fep_system = fep_system
        self.n_lambdas = len(self.fep_params.elmbs)
        self.config = config
        self.file = File(filename=self.config.filename)
        header, self.fmt = self.init_file_formatting()
        with self.file(method="w") as f:
            f.write_line(header)
        self.counter = 0
        LOG.debug("Successfully intialised FEPReporter")
        print("report interval", self.config.report_interval)

    def init_file_formatting(self) -> Tuple[str, str]:
        header = f"{'n':>10}"
        fmt = "{:>10}"
        for i in range(self.n_lambdas):
            header += f" {FEPReporter.HEADER.format(f'E{i}')}"
            fmt += f" {FEPReporter.FMT}"
        return header, fmt

    def describe_next_report(self, simulation: app.Simulation):
        steps = (
            self.config.report_interval
            - simulation.currentStep % self.config.report_interval
        )
        return (steps, True, False, True, True, None)

    def describeNextReport(self, *args):
        """
        Method for OpenMM
        """
        return self.describe_next_report(*args)

    def _report(self, sim: app.Simulation, state: mm.State) -> None:
        main_positions = state.getPositions()
        energies = []
        for elmb, slmb in zip(self.fep_params.elmbs, self.fep_params.slmbs):
            self.fep_system(elmb, slmb)
            try:
                self.dummy_context.setPositions(main_positions)
            except Exception as e:
                raise RuntimeError(
                    f"Could not set main positions into dummy context for lambdas e={elmb:.6f} s={slmb:.6f}"
                ) from e
            try:
                idx_energy = self.dummy_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoules_per_mole
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not evaulate energy of context e={elmb:.6f} s={slmb:.6f} in main positions"
                ) from e
            energies.append(idx_energy)
        try:
            line = self.fmt.format(self.counter, *energies)
        except Exception as e:
            raise RuntimeError("Could not format output for FEPReporter") from e
        with self.file(method="a") as f:
            f.write_line(line)
        self.counter += 1

    def report(self, *args) -> None:
        """
        Method for OpenMM
        """
        return self._report(*args)




class LambdaSchedule:
    """Manages lambda value schedules for FEP"""
    
    @staticmethod
    def get_default_schedules() -> Tuple[np.ndarray, np.ndarray]:
        """Get default electrostatic and steric lambda schedules"""
        # Electrostatic lambdas (turn off charges first)
        e_lambdas = np.array([
            1.00,0.96,0.91,0.87,0.83,
            0.78,0.74,0.70,0.65,0.61,
            0.57,0.52,0.48,0.43,0.39,
            0.35,0.30,0.26,0.22,0.17,
            0.13,0.09,0.04,0.00,0.00,
            0.00,0.00,0.00,0.00,0.00,
            0.00,0.00,0.00,0.00,0.00,
            0.00,0.00,0.00,0.00,0.00,
            0.00,0.00,0.00,0.00,0.00,
            0.00,0.00,
        ])
        
        # Steric lambdas (turn off VDW after charges)
        s_lambdas = np.array([
            1.00,1.00,1.00,1.00,1.00, 
            1.00,1.00,1.00,1.00,1.00,
            1.00,1.00,1.00,1.00,1.00,
            1.00,1.00,1.00,1.00,1.00,
            1.00,1.00,1.00,1.00,0.96,
            0.91,0.87,0.83,0.78,0.74,
            0.70,0.65,0.61,0.57,0.52,
            0.48,0.43,0.39,0.35,0.30,
            0.26,0.22,0.17,0.13,0.09,
            0.04,0.00,
        ])
        
        return e_lambdas, s_lambdas
    
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


            dummy_system, _, _ = system_builder_func()
            
            # Return systems for user to create simulations
            yield {
                'window_idx': window_idx,
                'elmb': elmb,
                'slmb': slmb,
                'system': system,
                'dummy_system': dummy_system,
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
    # gpu_id = 0
    
    # Platform
    platform = mm.Platform.getPlatformByName('HIP')
    # platform = mm.Platform.getPlatformByName('CPU')
    properties = {'Precision': 'mixed', 'DeviceIndex': str(0)}
    # properties = None
    
    # Thermodynamic state
    temperature = 300.0  # Kelvin
    pressure = 1.0  # atmospheres
    
    # Integration parameters
    timestep = 0.001  # picoseconds
    n_steps = 5000000
    trel = 1.0  # thermostat relaxation time (1/ps)
    
    # Output settings
    traj_frequency = 10000  # steps between trajectory frames
    thermo_frequency = 1000  # steps between thermodynamic outputs
   
    pdb = app.PDBFile(coord_file)
    residue_id_to_fep = 1
    atoms_to_fep = [int(a.index) for res in pdb.topology.residues() for a in res.atoms() if int(res.id) == residue_id_to_fep]
        
    e_lambdas, s_lambdas = LambdaSchedule.get_default_schedules()
    # e_lambdas = [1.0, 0.5, 0.0, 0.0, 0.0]
    # s_lambdas = [1.0, 1.0, 1.0, 0.5, 0.0]
        
    # FEP settings
    fep_params = FEPParams(
        elmbs=e_lambdas,
        slmbs=s_lambdas,
        atoms_to_fep=atoms_to_fep,  # Atom IDs to apply FEP
        atoms_to_restrain=[],  # Atom IDs to restrain
        m=2.0,
        n=1.0,
        alpha=0.5,
        restraint_k=14473.0  # kJ/mol/nm^2
    )

    fep_config = FEPConfig(filename = "fep.out", report_interval = 1000)
    
    
    # ========================================================================
    # LAMBDA SCHEDULE SETUP
    # ========================================================================
    
   
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

        og_fep_system = FEPSystem(simulation.context, fep_params)
        og_fep_system(window_data["elmb"], window_data["slmb"])
        
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
                10000,
                step=True,
                potentialEnergy=True,
                temperature=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=n_steps
            )
        )
        
        # Create neighbor simulations for energy evaluations
        contexts = [] 
        dummy_context = mm.app.Simulation(
                window_data['topology'],
                window_data['dummy_system'],
                mm.VerletIntegrator(1),
                platform,
                properties
            ).context
        dummy_context.setPositions(window_data['positions'])

        fep_system = FEPSystem(dummy_context, fep_params)

        simulation.reporters.append(
            FEPReporter(dummy_context, fep_params, fep_system, fep_config)
        )
        
        # Run simulation
        print(f"Running {n_steps} steps...")
        simulation.step(n_steps)
        
        # Save checkpoint
        simulation.saveCheckpoint('checkpoint.chk')
        
        print(f"Window {window_data['window_idx']} complete!\n")


if __name__ == "__main__":
    main()
