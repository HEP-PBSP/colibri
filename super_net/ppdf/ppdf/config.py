"""
ppdf.config.py

Config module of ppdf

Author: Mark N. Costantini
Date: 15.11.2023
"""

from super_net.config import SuperNetConfig, Environment


class Environment(Environment):
    pass


class PpdfConfig(SuperNetConfig):
    """
    WminConfig class Inherits from super_net.config.SuperNetConfig
    """

    def parse_pdf_prior(self, name):
        """PDF set used to generate prior values in grid fit"""
        return self.parse_pdf(name)
    