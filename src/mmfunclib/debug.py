import sys
import numpy as np
import openmm as mm
import openmm.unit as unit
import openmm.app as app
import logging

def debug_context_energy(
    c, outputUnit=unit.kilojoule_per_mole, header=None, eKin=False
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("debug")
    if header is None:
        logger.critical("#--- Single point energy ------------------#")
    else:
        logger.info(header)

    logger.info("  Context parameters ...")
    for p in c.getParameters():
        logger.info("    {:38s} = {} ".format(p, c.getParameter(p)))

    s = c.getSystem()
    e = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(outputUnit)

    symbol = outputUnit.get_symbol()
    logger.info("  {:40s} = {} ".format("Total potential energy (" + symbol + ")", e))

    for force in s.getForces():
        if isinstance(force, mm.CMMotionRemover):
            continue
        e = (
            c.getState(getEnergy=True, groups={force.getForceGroup()})
            .getPotentialEnergy()
            .value_in_unit(outputUnit)
        )
        logger.info("  {:40s} = {} ".format(force.getName() + " (" + symbol + ")", e))

    if eKin:
        k = c.getState(getEnergy=True).getKineticEnergy()
        if k.value_in_unit(outputUnit) > 1e-3:
            logger.info(
                "  {:40s} = {} ".format(
                    "Kinetic energy (" + symbol + ")", k.value_in_unit(outputUnit)
                )
            )
            temp = 2 * k / ndof / MOLAR_GAS_CONSTANT_R
            logger.info(
                "  {:40s} = {} ".format(
                    "Temperature (K)", temp.value_in_unit(unit.kelvin)
                )
            )

    return


