"""
grid_pdf.config.py

Config module of grid_pdf

Author: Mark N. Costantini
Date: 15.11.2023
"""
from reportengine.configparser import explicit_node

from super_net.config import SuperNetConfig, Environment
from super_net.utils import FLAVOUR_TO_ID_MAPPING
from grid_pdf import grid_pdf_commondata_utils

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
        lengths = [len(val) for (_,val) in xgrids.items()]
        # Remove all zero-length lists
        lengths = list(filter((0).__ne__, lengths))
        if not all(x == lengths[0] for x in lengths):
            raise ValueError("Cannot currently support reduced x-grids of different lengths.")
        return lengths[0]

    def produce_reduced_xgrids(self, xgrids):
        """The reduced x-grids used in the fit, organised by flavour."""
        return {FLAVOUR_TO_ID_MAPPING[flav] : val for (flav,val) in xgrids.items()}

    @explicit_node
    def produce_commondata_tuple(self, pseudodata=False, fakedata=False, some_condition_to_decided=None):
        """
        Note: this is needed so as to construct synthetic data (closure test data) using an
        interpolated grid.
        Hence there must be some logic so that this is only used for closure tests.
        """

        if some_condition_to_decided is not None:

            if pseudodata and fakedata:
                # closure test pseudodata
                return grid_pdf_commondata_utils.grid_pdf_closuretest_pseudodata_commondata_tuple
            
            elif fakedata:
                # closure test fake-data
                return grid_pdf_commondata_utils.grid_pdf_closuretest_commondata_tuple

        else:
            return super().produce_commondata_tuple(pseudodata=pseudodata, fakedata=fakedata)
        