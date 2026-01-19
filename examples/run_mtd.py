import sys
import numpy as np
import openmm as mm
import openmm.unit as unit
import openmm.app as app
import logging
import time
from dataclasses import dataclass
from typing import List

class CenterOfMassReporter:
    def __init__(self, file, reportInterval, atomIndices):
        """
        Parameters:
        - file: filename or file object to write to
        - reportInterval: number of steps between reports
        - atomIndices: list of atom indices for the molecule
        """
        self._out = open(file, 'w') if isinstance(file, str) else file
        self._reportInterval = reportInterval
        self._atomIndices = atomIndices
        self._hasInitialized = False

    def __del__(self):
        if hasattr(self, '_out'):
            self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        if not self._hasInitialized:
            self._out.write('Step COM_X COM_Y COM_Z\n')
            self._hasInitialized = True

        positions = state.getPositions(asNumpy=True).value_in_unit(mm.unit.nanometer)
        masses = [simulation.system.getParticleMass(i).value_in_unit(mm.unit.dalton)
                  for i in self._atomIndices]

        # Calculate center of mass
        total_mass = sum(masses)
        com = np.zeros(3)
        for i, idx in enumerate(self._atomIndices):
            com += masses[i] * positions[idx]
        com /= total_mass

        self._out.write(f'{simulation.currentStep:>16} {com[0]:>16.8f} {com[1]:>16.8f} {com[2]:>16.8f}\n')
        self._out.flush()

class ForceReporter:
    """Reporter that writes energy values for individual Force objects.

    Parameters
    ----------
    file : str or file-like object
        Output file path or file object
    reportInterval : int
        Frequency (in steps) at which to write data
    forces : list
        List of Force objects to track
    append : bool, optional
        If True, append to existing file. Default is False.
    """

    def __init__(self, file, reportInterval, forces, append=False):
        self._reportInterval = reportInterval
        self._forces = forces
        self._append = append
        self._hasWrittenHeader = False

        # Open file if a path is provided
        if isinstance(file, str):
            self._out = open(file, 'a' if append else 'w')
            self._ownFile = True
        else:
            self._out = file
            self._ownFile = False

    def describeNextReport(self, simulation):
        """Get information about the next report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            (steps, positions, velocities, forces, energies, groups)
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        # Request energy calculations for each force group
        return (steps, False, False, False, True, set(range(32)))

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        # Write header if first report
        if not self._hasWrittenHeader:
            self._writeHeader()
            self._hasWrittenHeader = True

        # Get step number
        step = simulation.currentStep

        # Write step
        self._out.write(f"{step}")

        # Write energy for each force
        for force in self._forces:
            group = force.getForceGroup()
            energy = simulation.context.getState(
                getEnergy=True, groups={group}
            ).getPotentialEnergy()
            # Convert to kJ/mol
            energy_value = energy.value_in_unit(unit.kilojoules_per_mole)
            self._out.write(f" {energy_value:12.6f}")

        self._out.write("\n")
        self._out.flush()

    def _writeHeader(self):
        """Write the header line with force names."""
        headers = ["Step"]

        for force in self._forces:
            # Get force name - use class name if no custom name
            force_name = force.__class__.__name__
            # Try to get a custom name if it exists
            try:
                if hasattr(force, 'getName') and force.getName():
                    force_name = force.getName()
            except:
                pass
            headers.append(f"{force_name:>12}")

        self._out.write(" ".join(headers) + "\n")
        self._out.flush()

    def __del__(self):
        """Close the file if we opened it."""
        if self._ownFile and hasattr(self, '_out'):
            self._out.close()


class HILLSReporter:

    def __init__(self, file, reportInterval, meta: app.Metadynamics):
        self._out = open(file, 'a')
        self._reportInterval = reportInterval
        self.meta = meta
        if self.meta.frequency != reportInterval:
            raise ValueError("HILL frequency must be same as Meta frequency")
        self.gaussian_widths = [v.biasWidth for v in self.meta.variables]
        self.hills_fmt = self.init_header()

    def init_header(self) -> str:
        n_cvs = len(self.meta.variables)
        str_cv = str_s = str_f1 = str_f2 = ""
        for i in range(n_cvs):
            str_cv += " cv" + str(i)
            str_s += " sigma_cv" + str(i)
            str_f1 += " {1[" + str(i) + "]:<16.8f}"
            str_f2 += " {" + str(i + 2) + ":<16.8f}"
        fmt_str = (
            "{0:<16.8f}"
            + str_f1
            + str_f2
            + " {"
            + str(n_cvs + 2)
            + ":<16.8f} {"
            + str(n_cvs + 3)
            + ":<16}\n"
        )
        self._out.write(f"#! FIELDS time{str_cv} {str_s} height biasf\n")
        return fmt_str


    def describeNextReport(self, simulation):
        """Get information about the next report this reporter will generate.
        
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
            
        Returns
        -------
        tuple
            A five element tuple: (steps, positions, velocities, forces, energies)
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, True, False, False)

    def report(self, simulation, state):
        
        self.meta._force.updateParametersInContext(simulation.context)

        position = self.meta._force.getCollectiveVariableValues(simulation.context)
        energy = simulation.context.getState(
            getEnergy=True, groups={self.meta._force.getForceGroup()}
        ).getPotentialEnergy()
        height = self.meta.height * np.exp(
            -energy / (unit.MOLAR_GAS_CONSTANT_R * self.meta._deltaT)
        )
        time = simulation.context.getTime().value_in_unit(unit.picosecond)
        scaled_height = (self.meta.biasFactor / (self.meta.biasFactor - 1)) * height
        self._out.write(
            self.hills_fmt.format(
                time, 
                position, 
                *self.gaussian_widths, 
                scaled_height._value, 
                self.meta.biasFactor
            )
        )
        self._out.flush()

    def __del__(self):
        if hasattr(self, '_out'):
            self._out.close()


@dataclass
class Grid:
    grid_min: float
    grid_max: float
    grid_width: float


def format_free_energy(dg: np.ndarray, grids: List[Grid], reduce_to_axis: int = 0):
    """
    Reduce N-dimensional free energy array to 1D along specified axis.
    
    Parameters:
    -----------
    dg : np.ndarray
        N-dimensional free energy array in kJ/mole
    grids : List[Grid]
        Grid parameters for each collective variable
    reduce_to_axis : int
        Axis to preserve (all others will be reduced by taking minimum)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns 'position' and 'free_energy'
    """
    if len(dg.shape) != len(grids):
        raise ValueError(f"Free energy array has dimension {len(dg.shape)} while grids for {len(grids)} were provided")
    
    if reduce_to_axis < 0 or reduce_to_axis >= len(grids):
        raise ValueError(f"reduce_to_axis must be between 0 and {len(grids)-1}")
    
    # Reduce array to 1D by taking minimum over all axes except the target
    axes_to_reduce = tuple(i for i in range(len(dg.shape)) if i != reduce_to_axis)
    dg_1d = np.min(dg, axis=axes_to_reduce)
    
    # Get grid parameters for the retained axis
    grid = grids[reduce_to_axis]
    
    # Calculate positions along the axis
    # Position i corresponds to: minValue + i*(maxValue-minValue)/gridWidth
    positions = np.array([
        grid.grid_min + i * (grid.grid_max - grid.grid_min) / grid.grid_width
        for i in range(int(grid.grid_width))
    ])
    
    # Create DataFrame
    df = pd.DataFrame({
        f'position(dim={reduce_to_axis})': positions,
        'free_energy': dg_1d
    })
    
    return df

def main():
    # Simulation parameters
    timestep = 0.001 * unit.picosecond
    n_steps = 50000000
    temperature = 300 * unit.kelvin
    thermostat_parameter = 1.0 / unit.picoseconds
    npt = False
    pressure = 1 * unit.bar
    barostat_update = 25
    n_screen = 1000
    n_traj = 5000
    n_file = 1000
    coordinates = sys.argv[1]

    task_id = str(sys.argv[3])

    time.sleep(int(task_id)) 

    # Read the coordinates
    pdb = app.PDBFile(coordinates)
    forcefield = app.ForceField(sys.argv[2])
    system = forcefield.createSystem(
        pdb.topology,
        #nonbondedMethod=app.NoCutoff,
        #nonbondedMethod=app.CutoffPeriodic,
        #nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedMethod=app.PME,  # type: ignore
        ewaldErrorTolerance=1e-05,
        nonbondedCutoff=0.9 * unit.nanometer,  # type: ignore
        useDispersionCorrection=False,
        constraints=None,
        rigidWater=False,
    )

    for n, f in enumerate(system.getForces()):
        f.setForceGroup(n + 1)

    # Create the integrator object
    integrator = mm.LangevinMiddleIntegrator(
        temperature, thermostat_parameter, timestep
    )

    # Add the barostat for NPT simulation
    if npt:
        system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_update))

    atoms_to_restrain = [14]

    restraint = mm.CustomExternalForce(
        "0.5*kr*periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    restraint.addGlobalParameter("kr", 
        14473 * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
   
    atoms = list(pdb.topology.atoms())
    for atom_id in atoms_to_restrain:
        atom = atoms[atom_id]
        pos = pdb.positions[atom.index]
        print(f"Restraining atom id={atom.id} index={atom.index} name={atom.name } at x={pos[0]} y={pos[1]} z={pos[2]}")
        restraint.addParticle(atom.index, pos)

    restraint.setName("PointRestraint")
    system.addForce(restraint)

    aspartate_residue_id = 1
    aspartate_atoms = [int(a.index) for res in pdb.topology.residues() for a in res.atoms() if int(res.id) == aspartate_residue_id]
    print("Aspartate atom indices: ", *aspartate_atoms) 

    com_reporter = CenterOfMassReporter(f"com-{task_id}.txt", 1000, aspartate_atoms)

    fixed_point = [
        2.980745 * unit.nanometer, 
        2.738263 * unit.nanometer, 
        4.544054 * unit.nanometer,
    ]

    com_force = mm.CustomCentroidBondForce(1,
        "z1-z0")
    com_force.addGlobalParameter("z0", fixed_point[2])
    group_idx = com_force.addGroup(aspartate_atoms)
    com_force.addBond([group_idx], [])
    # com_cv = mm.CustomCVForce('com_force')
    # com_cv.addCollectiveVariable('com_force', com_force)
    # test_cv = mm.CustomCVForce("0.5")

    upper_wall = 1.5 * unit.nanometer

    com_force_res = mm.CustomCentroidBondForce(1,
        "0.5 * kc * max(0, (z1-z0) - flat)^2")
    com_force_res.addGlobalParameter("z0", fixed_point[2])
    com_force_res.addGlobalParameter("flat", upper_wall)
    com_force_res.addGlobalParameter("kc", 
        14473 * unit.kilojoules_per_mole / unit.nanometer**2)
    group_idx2 = com_force_res.addGroup(aspartate_atoms)
    com_force_res.addBond([group_idx2], [])
    print(f"Upper wall on aspartate at {fixed_point[2] + upper_wall}")
    com_force_res.setName("UpperWall")
    system.addForce(com_force_res)

    # funnel/cylinder restraint
    com_force_funnel = mm.CustomCentroidBondForce(1,
#        "0.5 * kcyl * max(0, d - flatcyl)^2; d = (xij^2 + yij^2)^0.5; xij = x1-x0; yij = y1-y0")
        "0.5 * kcyl * max(0, d - r_funnel)^2; "
        "d = sqrt(xij^2 + yij^2); "
        "xij = x1 - x0; "
        "yij = y1 - y0; "
        "r_funnel = select(step(z1 - zstart), "  # if z >= z0
        "select(step(z1 - zstop), r_top, r_below + alpha * (z1 - zstart)), "  # funnel or top cylinder
        "r_below)"  # bottom cylinder        
    )
    com_force_funnel.addGlobalParameter("kcyl", 
        14473 * unit.kilojoules_per_mole / unit.nanometer**2)
    com_force_funnel.addGlobalParameter("x0", fixed_point[0])
    com_force_funnel.addGlobalParameter("y0", fixed_point[1])

    com_force_funnel.addGlobalParameter("zstart", zstart := fixed_point[2])   # z where funnel starts widening
    com_force_funnel.addGlobalParameter("zstop", zstop := fixed_point[2] + 0.6 * unit.nanometer)    # z where funnel stops (top cylinder starts)
    com_force_funnel.addGlobalParameter("r_below", r_below := 0.15 * unit.nanometer)  # radius below zstart
    com_force_funnel.addGlobalParameter("r_top", r_top := 0.5 * unit.nanometer)    # radius above zstop
    alpha = (r_top - r_below) / (zstop - zstart)
    com_force_funnel.addGlobalParameter("alpha", alpha)    # rate of radius increase in funnel
    group_idx3 = com_force_funnel.addGroup(aspartate_atoms)
    com_force_funnel.addBond([group_idx3], [])
    print(f"Cylindrical/funnel restraint on aspartate at {fixed_point[0]} {fixed_point[1]}:")
    print(f"  z < {zstart}: Cylindrical restraint with radius = {r_below}")
    print(f"  {zstart} ≤ z < {zstop}: Funnel that widens linearly from {r_below} to {r_top}")
    print(f"  z ≥ {zstop}: Cylindrical restraint with radius = {r_top}")
    com_force_funnel.setName("Cylinder+Funnel")
    system.addForce(com_force_funnel)

    for n, f in enumerate(system.getForces()):
        f.setForceGroup(n + 1)
    restraint_forces = [restraint, com_force_res, com_force_funnel]

    bv = app.BiasVariable(
        com_force, 
        bv0_grid_min := -0.5 * unit.nanometer, 
        bv0_grid_max := 1.8 * unit.nanometer, 
        bv0_grid_width := 0.01 * unit.nanometer, 
        False
    )

    meta = app.Metadynamics(
        system, 
        [bv], 
        temperature, 
        bf := 22, 
        height := 2.494 * unit.kilojoule_per_mole, 
        freq := 1000, 
        save_freq := freq, 
        bias_dir := "./BIAS"
    )

    hills = HILLSReporter(f"./BIAS/HILLS.{task_id}", freq, meta)

    # Create the simulation object
    simulation = app.Simulation(
        pdb.topology,
        system,
        integrator,
        # mm.Platform.getPlatformByName("CPU"),
        # mm.Platform.getPlatformByName("OpenCL"),
        # mm.Platform.getPlatformByName("CUDA"),
        mm.Platform.getPlatformByName("HIP"),
        {"Precision": "mixed"},
    )

    # Add the velocities to the simulation
    simulation.context.setPositions(pdb.positions)

    # Screen output
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            n_screen,
            totalSteps=int(n_steps),
            separator="\t",
            step=False,
            time=True,
            potentialEnergy=True,
            kineticEnergy=False,
            totalEnergy=False,
            temperature=True,
            volume=False,
            density=True,
            progress=True,
            remainingTime=True,
            speed=True,
            elapsedTime=False,
        )
    )

    # File output
    simulation.reporters.append(
        app.StateDataReporter(
            f"md-{task_id}.log",
            n_file,
            separator=",",
            step=False,
            time=True,
            potentialEnergy=True,
            kineticEnergy=False,
            totalEnergy=False,
            temperature=True,
            volume=True,
            density=True,
            progress=False,
            remainingTime=False,
            speed=False,
            elapsedTime=False,
        )
    )

    # Trajectory output
    simulation.reporters.append(app.DCDReporter(f"traj-{task_id}.dcd", n_traj))

    # Checkpoint reporter
    simulation.reporters.append(app.CheckpointReporter(f"checkpoint-{task_id}.chk", 50000, writeState=False))

    simulation.reporters.append(hills)

    simulation.reporters.append(com_reporter)

    simulation.reporters.append(
        ForceReporter(f'force_energies-{task_id}.txt', 1000, restraint_forces)
    )

    # # Energy minimisation
    # e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    # print("Initial energy: %s", e)
    # simulation.minimizeEnergy(tolerance=0.001)
    # e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    # print("Energy after minimisation: %s", e)
    # positions = simulation.context.getState(getPositions=True).getPositions()
    # simulation.context.reinitialize()
    # simulation.context.setPositions(positions)
    # app.PDBFile.writeFile(simulation.topology, positions, "opti.pdb")

    # Generate the velocities for the simulation
    r = np.random.randint(1, 99999)
    simulation.context.setVelocitiesToTemperature(temperature, r)

    bv0_grid = Grid(bv0_grid_min, bv0_grid_max, bv0_grid_width)

    # Run molecular dynamics
    inc = 1000
    for s in range(int(n_steps) // inc):
        meta.step(simulation, inc)
        cvs = meta.getCollectiveVariables(simulation)
        if (s % 10) == 0 and task_id == "000":
            # np.savetxt(f"free_energy.txt", meta.getFreeEnergy())
            dgdf = format_free_energy(meta.getFreeEnergy(), [bv0_grid], 0)
            dgdf.to_csv("free_energy.csv", sep="\\s+")

    np.savetxt(f"free_energy.txt", meta.getFreeEnergy())

    # Save checkpoint
    simulation.saveCheckpoint(f'checkpoint-{task_id}.chk')


if __name__ == "__main__":
    main()
