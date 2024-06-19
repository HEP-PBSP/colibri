"""
grid_pdf.config.py

Config module of grid_pdf

"""

import dill
from grid_pdf.model import GridPDFModel

from colibri.config import Environment, colibriConfig


class Environment(Environment):
    pass


class GridPdfConfig(colibriConfig):
    """
    GridConfig class Inherits from colibri.config.colibriConfig
    """

    def produce_flavour_xgrids(self, grid_pdf_settings):
        return grid_pdf_settings["xgrids"]

    def parse_prior_settings(self, settings):
        # In the case of a uniform prior around a central PDF, we also need
        # to specify the total number of standard deviations around the mean
        # which we allow.
        if settings["type"] == "uniform_pdf_prior":
            # Check if number of standard deviations are supplied, default to 2
            # otherwise.
            if "nsigma" not in settings.keys():
                settings["nsigma"] = 2

        return settings

    def produce_pdf_model(self, flavour_xgrids, output_path, float_type=None):
        """
        Produce the PDF model for the grid_pdf fit.
        """
        model = GridPDFModel(flavour_xgrids, float_type=float_type)
        # dump model to output_path using dill
        # this is mainly needed by scripts/ns_resampler.py
        with open(output_path / "pdf_model.pkl", "wb") as file:
            dill.dump(model, file)
        return model
