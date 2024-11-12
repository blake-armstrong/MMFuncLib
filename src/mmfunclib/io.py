import logging
from dataclasses import dataclass
from typing import Optional

LOG = logging.getLogger(__name__)


class File:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.file = None
        self.method = "w"

    def __call__(self, method: str = "w"):
        self.method = str(method)
        return self

    def __enter__(self):
        self.file = open(self.filename, self.method, encoding="utf-8")
        LOG.debug("Sucessfully writing to file: '%s'", self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.file:
            return
        LOG.debug("Finished writing to file: '%s'", self.filename)
        self.file.close()
        LOG.debug("Closing file: '%s'", self.filename)

    def write_line(self, text, flush=True):
        if not self.file:
            return
        self.file.write(text + "\n")
        if flush:
            self.file.flush()

    def get_last_line(self) -> Optional[str]:
        if not self.file:
            return None
        lines = self.file.readlines()
        if not lines:
            raise RuntimeError(f"Could not read from file: '{self.filename}")
        return lines[-1]


@dataclass
class Config:

    def __post_init__(self):
        LOG.info("%s parameters:", self.__class__.__name__)
        attr = vars(self)
        for k, v in attr.items():
            LOG.info("  %s = %s", k, v)
