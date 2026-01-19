#!/usr/bin/env python
"""
MBAR analysis script for FEP output files
Reads fep.out files from lambda_??? directories and computes free energy differences
"""

import numpy as np
import pandas as pd
import pymbar
import openmm.unit as unit
import os
import glob

def add_units():
    """Add custom electron volt units for MD simulations"""
    r0 = (1 * unit.elementary_charge * unit.volt).value_in_unit(unit.joule)
    r1 = (1 * unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
        unit.kilojoule_per_mole
    )

    ev_base_unit = unit.ScaledUnit(r0, unit.kilojoule_per_mole, "electron_volt", "eV")
    unit.ev_base_unit = ev_base_unit
    unit.ev = unit.electron_volt = unit.Unit({ev_base_unit: 1.0})

    md_ev_base_unit = unit.ScaledUnit(r1, unit.kilojoule_per_mole, "electron_volt", "eV")
    unit.md_ev_base_unit = md_ev_base_unit
    unit.md_ev = unit.md_electron_volt = unit.Unit({md_ev_base_unit: 1.0})

def extract_lambda_number(dir_path):
    """Extract the 3-digit number from lambda_??? directory name"""
    return int(dir_path.rstrip('/').split('_')[1])

def read_fep_output(filepath):
    """
    Read a fep.out file and extract the energy matrix
    
    Parameters
    ----------
    filepath : str
        Path to the fep.out file
        
    Returns
    -------
    energies : np.ndarray
        2D array of energies (n_samples, n_states)
    lambda_cols : list
        List of lambda column names (E0, E1, E2, ...)
    """
    # Read the file, skipping comment lines
    # The 'n' column contains row indices
    df = pd.read_csv(filepath, sep=r'\s+', comment='#')
    
    # Extract columns that start with 'E' (energy columns)
    lambda_cols = [col for col in df.columns if col.startswith('E')]
    
    if len(lambda_cols) == 0:
        raise ValueError(f"No energy columns found in {filepath}")
    
    # Extract energy matrix (only the E0, E1, E2, ... columns)
    energies = df[lambda_cols].values

    return energies, lambda_cols, df

def analyse_fep_mbar(fep_dir, temperature=300.0, units='omm', charge=0.0, 
                     dielectric=79.988, cell=2.5, skip=50, equil=1, 
                     tolerance=1e-7, max_iterations=10000):
    """
    Perform MBAR analysis on FEP output files
    
    Parameters
    ----------
    fep_dir : str
        Path to the FEP directory containing lambda_??? subdirectories
    temperature : float
        Temperature in Kelvin (default: 300 K)
    units : str
        Unit system: 'omm' for OpenMM or 'lmp' for LAMMPS (default: 'omm')
    charge : float
        Net charge for finite-size correction (default: 0.0)
    dielectric : float
        Dielectric constant (default: 79.988 for water)
    cell : float
        Cell size in nanometers (omm) or angstroms (lmp) (default: 2.5)
    skip : int
        Number of equilibration frames to skip (default: 50)
    equil : int
        Skip every N frames after equilibration (default: 1)
    tolerance : float
        Convergence tolerance for MBAR solver (default: 1e-7)
    max_iterations : int
        Maximum iterations for MBAR solver (default: 10000)
        
    Returns
    -------
    results : dict
        Dictionary containing MBAR results
    """
    add_units()
    
    if units == "omm":
        unit_system = {"d": unit.nanometer, "e": unit.kilojoule_per_mole}
    elif units == "lmp":
        unit_system = {"d": unit.angstrom, "e": unit.electron_volt}
    else:
        raise ValueError("Units must be 'omm' or 'lmp'")
    
    unit_system["t"] = unit.kelvin
    unit_system["q"] = unit.elementary_charge
    
    cell_with_unit = cell * unit_system["d"]
    temp_with_unit = temperature * unit.kelvin
    charge_with_unit = charge * unit.elementary_charge
    
    kT = (unit.MOLAR_GAS_CONSTANT_R * temp_with_unit).in_units_of(unit.kilojoule_per_mole)
    beta = 1.0 / kT.value_in_unit(unit_system["e"])

    CONST_pi4e0 = (
        14.399645 * unit.md_electron_volt * unit.angstrom / unit.elementary_charge**2
    )
    pi4e0 = CONST_pi4e0.in_units_of(
        unit_system["e"] * unit_system["d"] / unit.elementary_charge**2
    )
    zeta = 2.837297
    
    qcorr = (
        pi4e0 * zeta * 0.5 * charge_with_unit ** 2 / dielectric / cell_with_unit
    ).value_in_unit(unit.kilojoule_per_mole)
    
    string = [x.get_symbol() for x in unit_system.values()]
    print("Units         = {}".format(string))
    print("Temperature   = {}".format(temp_with_unit))
    print("Charge        = {}".format(charge_with_unit))
    print("Dielectric    = {:.3f}".format(dielectric))
    print("Cell size     = {}".format(cell_with_unit))
    print("Equilibration = {} frames".format(skip))
    print("Skip data     = every {} frames".format(equil))
    print()
    
    lambda_dirs = sorted(glob.glob(os.path.join(fep_dir, 'lambda_*/')), 
                         key=extract_lambda_number)
    n_states = len(lambda_dirs)
    
    if n_states == 0:
        raise ValueError(f"No lambda_* directories found in {fep_dir}")
    
    print(f"Found {n_states} lambda states")

    u_kn_list = []
    N_k = np.zeros(n_states, dtype=int)
    
    for k, lambda_dir in enumerate(lambda_dirs):
        fep_file = os.path.join(lambda_dir, 'fep.out')
        
        if not os.path.exists(fep_file):
            raise ValueError(f"File not found: {fep_file}")
        
        lambda_num = extract_lambda_number(lambda_dir)
        print(f"Reading lambda_{lambda_num:03d}...")
        
        energies, lambda_cols, df = read_fep_output(fep_file)
        
        if energies.shape[1] != n_states:
            raise ValueError(f"Expected {n_states} energy columns, got {energies.shape[1]}")
        
        mask = df['n'] > skip
        energies_processed = energies[mask][::equil]
        energies_processed = energies_processed.T * beta
        
        u_kn_list.append(energies_processed)
        N_k[k] = energies_processed.shape[1]
        
    n_total = N_k.sum()
    u_kn = np.concatenate(u_kn_list, axis=1)
    
    print("\nInitialising MBAR...")
    mbar = pymbar.MBAR(u_kn, N_k, verbose=True, initialize="BAR")
    
    print("\nComputing free energy differences...")
    results = mbar.compute_free_energy_differences()
    
    # Extract results
    delta_f = results['Delta_f']  # Free energy differences (in kT units)
    d_delta_f = results['dDelta_f']  # Uncertainties
    
    delta_f_kj = delta_f * kT.value_in_unit(unit.kilojoule_per_mole)
    d_delta_f_kj = d_delta_f * kT.value_in_unit(unit.kilojoule_per_mole)
    
    print("\n" + "="*60)
    print("MBAR RESULTS")
    print("="*60)
    print(f"\nFree energy differences (kJ/mol) relative to state 0:")
    print(f"{'State':<8} {'ΔF (kJ/mol)':<15} {'dΔF (kJ/mol)':<15}")
    print("-"*50)
    for i in range(n_states):
        print(f"{i:<8} {delta_f_kj[0, i]:<15.4f} {d_delta_f_kj[0, i]:<15.4f}")
    
    print(f"\nTotal free energy change (state 0 -> state {n_states-1}):")
    print(f"ΔF = {delta_f_kj[0, -1]:.4f} ± {d_delta_f_kj[0, -1]:.4f} kJ/mol")
    
    if abs(charge) > 1e-6:
        print(f"\nCharge correction = {qcorr:.4f} kJ/mol")
        print(f"Corrected ΔF = {delta_f_kj[0, -1] + qcorr:.4f} ± {d_delta_f_kj[0, -1]:.4f} kJ/mol")
    
    print("="*60)
    
    results_df = pd.DataFrame({
        'State': range(n_states),
        'Delta_F_kJ': delta_f_kj[0, :],
        'dDelta_F_kJ': d_delta_f_kj[0, :],
        'N_samples': N_k
    })
    
    return {
        'mbar': mbar,
        'delta_f': delta_f,
        'd_delta_f': d_delta_f,
        'delta_f_kj': delta_f_kj,
        'd_delta_f_kj': d_delta_f_kj,
        'N_k': N_k,
        'u_kn': u_kn,
        'n_states': n_states,
        'results_df': results_df,
        'charge_correction': qcorr,
        'kT': kT.value_in_unit(unit.kilojoule_per_mole)
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyse FEP output with MBAR',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('fep_dir', type=str, nargs='?', default='FEP',
                        help='Path to FEP directory (default: FEP)')
    parser.add_argument('--temperature', '-t', type=float, default=300.0, 
                        help='Temperature in Kelvin (default: 300 K)')
    parser.add_argument('--units', '-u', type=str, default='omm',
                        choices=['omm', 'lmp'],
                        help='Unit system: omm (OpenMM) or lmp (LAMMPS) (default: omm)')
    parser.add_argument('--charge', '-q', type=float, default=0.0,
                        help='Net charge for finite-size correction (default: 0.0)')
    parser.add_argument('--dielec', '-de', type=float, default=79.988,
                        help='Dielectric constant (default: 79.988)')
    parser.add_argument('--cell', '-l', type=float, default=2.5,
                        help='Cell size in nm (omm) or Å (lmp) (default: 2.5)')
    parser.add_argument('--skip', '-s', type=int, default=50,
                        help='Number of equilibration frames to skip (default: 50)')
    parser.add_argument('--equil', '-e', type=int, default=1,
                        help='Skip every N frames after equilibration (default: 1)')
    parser.add_argument('--tolerance', type=float, default=1e-7,
                        help='MBAR convergence tolerance (default: 1e-7, try 1e-5 or 1e-4 if convergence fails)')
    parser.add_argument('--max-iterations', type=int, default=10000,
                        help='Maximum MBAR iterations (default: 10000)')
    parser.add_argument('--output', type=str, default='mbar_results',
                        help='Output file prefix (default: mbar_results)')
    
    args = parser.parse_args()
    
    # Run MBAR analysis
    results = analyse_fep_mbar(
        args.fep_dir,
        temperature=args.temperature,
        units=args.units,
        charge=args.charge,
        dielectric=args.dielec,
        cell=args.cell,
        skip=args.skip,
        equil=args.equil,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations
    )
    
    # Save results
    np.savez(f"{args.output}.npz",
             delta_f=results['delta_f'],
             d_delta_f=results['d_delta_f'],
             delta_f_kj=results['delta_f_kj'],
             d_delta_f_kj=results['d_delta_f_kj'],
             N_k=results['N_k'],
             n_states=results['n_states'],
             charge_correction=results['charge_correction'],
             kT=results['kT'])
    
    # Save DataFrame as CSV
    results['results_df'].to_csv(f"{args.output}.csv", index=False)
    
    print(f"\nResults saved to {args.output}.npz and {args.output}.csv")
