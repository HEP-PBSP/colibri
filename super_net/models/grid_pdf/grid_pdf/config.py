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
        return grid_pdf_settings['xgrids']

    def produce_grid_prior(self, grid_pdf_settings):
        if 'prior_settings' in grid_pdf_settings.keys():
            settings = grid_pdf_settings['prior_settings']
            settings['pdf_prior'] = self.parse_pdf(settings['pdf_prior'])
            return settings
        return None

    def produce_grid_initialiser(self, grid_pdf_settings):
        if 'initialiser_settings' in grid_pdf_settings.keys():
            settings = grid_pdf_settings['initialiser_settings']
            settings['pdf_prior'] = self.parse_pdf(settings['pdf_prior'])
            return settings
        return None
