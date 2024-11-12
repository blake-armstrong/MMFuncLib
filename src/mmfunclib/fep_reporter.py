import logging
from typing import List, Tuple
from dataclasses import dataclass

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from .io import File, Config

LOG = logging.getLogger(__name__)


@dataclass
class FEPConfig(Config):
    filename: str = "fep.out"
    report_interval: int = 1000


DEFAULT_CONFIG = FEPConfig()


class FEPReporter:
    FMT = "{:20.8}"
    HEADER = "{:>20}"

    def __init__(
        self, contexts: List[mm.Context], config: FEPConfig = DEFAULT_CONFIG
    ) -> None:
        LOG.debug("Intialising FEPReporter")
        self.contexts = contexts
        self.n_contexts = len(contexts)
        self.config = config
        self.file = File(filename=self.config.filename)
        header, self.fmt = self.init_file_formatting()
        with self.file(method="w") as f:
            f.write_line(header)
        self.counter = 0
        LOG.debug("Successfully intialised FEPReporter")

    def init_file_formatting(self) -> Tuple[str, str]:
        header = f"{'n':>10}"
        fmt = "{:>10}"
        for i in range(self.n_contexts):
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

    def _report(self, _: app.Simulation, state: mm.State) -> None:
        main_positions = state.getPositions()
        main_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        energies = [main_energy]
        for idx, context in enumerate(self.contexts):
            try:
                context.setPositions(main_positions)
            except Exception as e:
                raise RuntimeError(
                    f"Could not set main positions into context {idx}"
                ) from e
            try:
                idx_energy = state.getPotentialEnergy().value_in_unit(
                    unit.kilojoules_per_mole
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not evaulate energy of context {idx} in main positions"
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
