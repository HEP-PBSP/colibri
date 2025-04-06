"""
les_houches.config.py

"""

import dill
import logging
from les_houches_example.model import ExamplePDFModel

from colibri.config import Environment, colibriConfig
from colibri.constants import FLAVOUR_TO_ID_MAPPING


log = logging.getLogger(__name__)


class Environment(Environment):
    pass


class ExampleConfig(colibriConfig):
    """
    GPConfig class Inherits from colibri.config.colibriConfig
    """

    def produce_fitted_flavours(self, flavour_mapping):
        return flavour_mapping

    def produce_pdf_model(self, output_path, fitted_flavours):
        """
        Produce the example PDF model.
        """
        model = ExamplePDFModel(fitted_flavours)
        # dump model to output_path using dill
        # this is mainly needed by scripts/ns_resampler.py
        with open(output_path / "pdf_model.pkl", "wb") as file:
            dill.dump(model, file)
        return model
