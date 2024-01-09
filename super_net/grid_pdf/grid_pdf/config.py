"""
grid_pdf.config.py

Config module of grid_pdf

Author: Mark N. Costantini
Date: 15.11.2023
"""

from super_net.config import SuperNetConfig, Environment
from super_net.utils import FLAVOUR_TO_ID_MAPPING


class Environment(Environment):
    pass


class GridPdfConfig(SuperNetConfig):
    """
    GridConfig class Inherits from super_net.config.SuperNetConfig
    """

    def parse_pdf_prior(self, name):
        """PDF set used to generate prior values in grid fit"""
        return self.parse_pdf(name)

    def produce_length_reduced_xgrids(self, xgrids):
        """The reduced x-grids used in the fit, organised by flavour."""
        lengths = [len(val) for (_, val) in xgrids.items()]
        # Remove all zero-length lists
        lengths = list(filter((0).__ne__, lengths))
        if not all(x == lengths[0] for x in lengths):
            raise ValueError(
                "Cannot currently support reduced x-grids of different lengths."
            )
        return lengths[0]

    def produce_length_stackedpdf(self, xgrids):
        """The lenght of the stacked PDF."""
        stack = []
        for _, val in xgrids.items():
            stack += val

        return len(stack)

    def produce_reduced_xgrids(self, xgrids):
        """The reduced x-grids used in the fit, organised by flavour."""
        return {FLAVOUR_TO_ID_MAPPING[flav]: val for (flav, val) in xgrids.items()}

