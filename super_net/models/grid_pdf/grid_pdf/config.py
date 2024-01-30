"""
grid_pdf.config.py

Config module of grid_pdf

Author: Mark N. Costantini
Date: 15.11.2023
"""

from super_net.config import SuperNetConfig, Environment


class Environment(Environment):
    pass


class GridPdfConfig(SuperNetConfig):
    """
    GridConfig class Inherits from super_net.config.SuperNetConfig
    """

    def produce_flavour_xgrids(self, grid_pdf_settings):
        return grid_pdf_settings["xgrids"]

    def parse_prior_settings(self, settings):
        # Currently, all possible prior choices require a central PDF.
        if "pdf_prior" not in settings.keys():
            raise ValueError("Missing key prior_pdf for uniform_pdf_prior")

        # In the case of a uniform prior around a central PDF, we also need
        # to specify the total number of standard deviations around the mean
        # which we allow.
        if settings["type"] == "uniform_pdf_prior":
            # Check if number of standard deviations are supplied, default to 2
            # otherwise.
            if "nsigma" not in settings.keys():
                settings["nsigma"] = 2

        return settings
