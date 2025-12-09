import os
import glob
import numpy as np
import pandas as pd
import openmm.unit as unit
import pymbar as mb
import argparse


def extract_lambda_number(dir_path):
    """Extract the 3-digit number from lambda_??? directory name"""
    # Remove trailing slash and get the last 3 characters
    return int(dir_path.rstrip('/').split('_')[1])


DEFAULTS = {
    "units": "omm",
    "temp": 300,
    "charge": 0.0,
    "dielectric": 79.988,
    "cell": 2.5,
    "skip": 1,
    "equil": 50,
    "debug": True,
}

def add_units():
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

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--charge", "-q", dest="charge", type=float)
    parser.add_argument("--temperature", "-t", dest="temp", type=float)
    parser.add_argument("--dielec", "-de", dest="dielectric", type=float)
    parser.add_argument("--cell", "-l", dest="cell", type=float)
    parser.add_argument("--units", "-u", dest="units")
    parser.add_argument(
        "--debug", "-v", dest="debug", action=argparse.BooleanOptionalAction
    )
    
    parser.add_argument("--skip", "-s", dest="skip", type=int)
    parser.add_argument("--equil", "-e", dest="equil", type=int)
    
    args = parser.parse_args()
    params = DEFAULTS.copy()
    for key, value in vars(args).items():
        if value is None:
            continue
        if key in params:
            params[key] = value

    add_units()

    if params["units"] == "omm":
        params["units"] = {"d": unit.nanometer, "e": unit.kilojoule_per_mole}
    elif params["units"] == "lmp":
        params["units"] = {"d": unit.angstrom, "e": unit.electron_volt}
    else:
        raise Exception("Units must be specified with -omm/-lmp")

    params["units"]["t"] = unit.kelvin
    params["units"]["q"] = unit.elementary_charge

    params["cell"] *= params["units"]["d"]
    params["temp"] *= unit.kelvin
    params["charge"] *= unit.elementary_charge

    return params


def main():
    folder = "FEP"
    if not os.path.isdir(folder): 
        raise ValueError(f"No {folder} folder!")

    directories = glob.glob(f"{folder}/lambda_???/")
    directories.sort(key=extract_lambda_number)

    total_stages = len(directories)
    print(f'{total_stages} total stages')

    params = parse_args()

    CONST_pi4e0 = (
        14.399645 * unit.md_electron_volt * unit.angstrom / unit.elementary_charge**2
    )
    pi4e0 = CONST_pi4e0.in_units_of(
        params["units"]["e"] * params["units"]["d"] / unit.elementary_charge**2
    )
    kT = (unit.MOLAR_GAS_CONSTANT_R * params["temp"]).in_units_of(
        unit.kilojoule_per_mole
    )
    zeta = 2.837297

    string = [x.get_symbol() for x in params["units"].values()]
    print("Units         = {}".format(string))
    print("Temperature   = {}".format(temperature := params["temp"]))
    print("Charge        = {}".format(charge := params["charge"]))
    print("Dielectic     = {:.3f}".format(dielectric := params["dielectric"]))
    print("Cell size     = {}".format(cell := params["cell"]))
    print("Equilibration = {}".format(skip := int(params["equil"])))
    print("Skip data     = {}".format(slice := int(params["skip"])))

    accumulated_free_energy = {
       "total": {"fwd": 0.0, "bar": 0.0, "bwd": 0.0},
       "error": {"fwd": 0.0, "bar": 0.0, "bwd": 0.0},
    }

    for n, directory in enumerate(directories):
        if n == total_stages - 1:
            continue
        lambda_num = extract_lambda_number(directory)
       
        # Check if fep.out exists
        fep_out_path = os.path.join(directory, "fep.out")
    
        if not os.path.exists(fep_out_path):
            raise RuntimeError(f"fep.out not found in {directory}")
        
        df_current = pd.read_csv(fep_out_path, sep="\\s+")

        head = "E2"
        if n == 0:
            head = "E1"
        forward_dE = df_current[head] - df_current["E0"]
        forward_dE = forward_dE[df_current["n"] > skip][::slice] / kT
        forward_dG = mb.other_estimators.exp(forward_dE, is_timeseries = True)

        # Check if fep.out exists
        fep_out_path = os.path.join(directories[n + 1], "fep.out")
    
        if not os.path.exists(fep_out_path):
            raise RuntimeError(f"fep.out not found in {directory}")
        
        df_forward = pd.read_csv(fep_out_path, sep="\\s+")

        backward_dE = df_forward["E1"] - df_forward["E0"]
        backward_dE = backward_dE[df_forward["n"] > skip][::slice] / kT
        backward_dG = mb.other_estimators.exp(backward_dE, is_timeseries = True)

        bar = mb.other_estimators.bar(forward_dE, backward_dE, uncertainty_method="MBAR")

        accumulated_free_energy["total"]["fwd"] += forward_dG["Delta_f"]
        accumulated_free_energy["error"]["fwd"] += forward_dG["dDelta_f"] ** 2
        accumulated_free_energy["total"]["bwd"] += backward_dG["Delta_f"]
        accumulated_free_energy["error"]["bwd"] += backward_dG["dDelta_f"] ** 2
        accumulated_free_energy["total"]["bar"] += bar["Delta_f"]
        accumulated_free_energy["error"]["bar"] += bar["dDelta_f"] ** 2

        dg = np.array([
            forward_dG["Delta_f"],
            forward_dG["dDelta_f"] ** 2,
            bar["Delta_f"],
            bar["dDelta_f"] ** 2,
           -backward_dG["Delta_f"],
            backward_dG["dDelta_f"] ** 2,
        ])
        stage = extract_lambda_number(directories[n])
        print(
            "dG ({:3}) fwd : {:7.2f} +/- {:.2f} | bar : {:7.2f} +/- {:.2f} | bwd : {:7.2f} +/- {:.2f}".format(
                stage, *dg
            )
        )
    print(
        "Total dG  = {:.3f} +/- {:.3f} kJ/mol".format(
            accumulated_free_energy["total"]["bar"] * kT.value_in_unit(unit.kilojoule_per_mole),
            accumulated_free_energy["error"]["bar"] * kT.value_in_unit(unit.kilojoule_per_mole)
        )
    )
    qcorr = (
        pi4e0 * zeta * 0.5 * charge ** 2 / dielectric / cell
    ).value_in_unit(unit.kilojoule_per_mole)
    print("Charge correction = {:.3f} kJ/mol".format(qcorr))
    print("#---------------------------------------------------------#")

    if params["debug"]:
        out = ["fwd", "bar", "bwd"]
    else:
        out = ["bar"]

    for idx in out:
        res1 = accumulated_free_energy["total"][idx] * kT.value_in_unit(unit.kilojoule_per_mole) + qcorr
        err1 = accumulated_free_energy["error"][idx] * kT.value_in_unit(unit.kilojoule_per_mole)
        print(
            "Total corrected solvation free energy ({}) = {:8.3f} +/- {:.3f} kJ/mol ".format(
                idx, res1, err1
            )
        )



main()
