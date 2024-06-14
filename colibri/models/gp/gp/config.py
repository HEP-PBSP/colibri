"""
grid_pdf.config.py

Config module of grid_pdf

"""

import dill
import logging
from gp.model import GpPDFModel

from colibri.config import Environment, colibriConfig
from colibri.constants import XGRID
from validphys import convolution

log = logging.getLogger(__name__)


class Environment(Environment):
    pass


class GPConfig(colibriConfig):
    """
    GPConfig class Inherits from colibri.config.colibriConfig
    """

    def produce_gp_xgrid(self, custom_xgrid=XGRID):
        """
        Produce the xgrid for the Gaussian process.

        Parameters
        ----------
        gp_xgrid: list
            The xgrid for the Gaussian process. Default is XGRID.
        """
        if custom_xgrid != XGRID:
            log.warning("Using custom xgrid for the Gaussian process.")
        return custom_xgrid

    def parse_gp_hyperparams_settings(self, gp_hyperparams_settings={}):
        """
        Parse the hyperparameters settings for the Gaussian process.

        Parameters
        ----------
        gp_hyperparams_settings: dict
            The hyperparameters settings for the Gaussian process.
            gp_hyperparams_settings should be defined in runcard.
        """
        return gp_hyperparams_settings

    def produce_fitted_flavours(self, flavour_mapping=[]):
        """The fitted flavours used in the model, in STANDARDISED order,
        according to convolution.FK_FLAVOURS.
        """
        if flavour_mapping == []:
            log.warning(
                f"No flavour mapping provided. Using all {len(convolution.FK_FLAVOURS)} flavours."
            )
            return convolution.FK_FLAVOURS

        flavours = []
        for flavour in convolution.FK_FLAVOURS:
            if flavour in flavour_mapping:
                flavours += [flavour]

        log.info(f"Flavours included in the fit: {flavours}")
        return flavours

    def produce_pdf_model(
        self, gp_xgrid, fitted_flavours, gp_hyperparams_settings, output_path
    ):
        """
        Produce the PDF model for the Gaussian process fit.
        """
        model = GpPDFModel(gp_xgrid, fitted_flavours, gp_hyperparams_settings)
        # dump model to output_path using dill
        # this is mainly needed by scripts/ns_resampler.py
        with open(output_path / "pdf_model.pkl", "wb") as file:
            dill.dump(model, file)
        return model
