"""
grid_pdf.config.py

Config module of grid_pdf

"""

import dill
from gp.model import GpPDFModel

from colibri.config import Environment, colibriConfig


class Environment(Environment):
    pass


class GPConfig(colibriConfig):
    """
    GPConfig class Inherits from colibri.config.colibriConfig
    """

    def produce_flavour_xgrids(self, grid_pdf_settings):
        return grid_pdf_settings["xgrids"]

    def produce_pdf_model(self, flavour_xgrids, output_path):
        """
        Produce the PDF model for the grid_pdf fit.
        """
        model = GpPDFModel(flavour_xgrids)
        # dump model to output_path using dill
        # this is mainly needed by scripts/ns_resampler.py
        with open(output_path / "pdf_model.pkl", "wb") as file:
            dill.dump(model, file)
        return model
