"""
les_houches_example.config.py

"""

import dill
import logging
from les_houches_example.model import LesHouchesPDF

from colibri.config import Environment, colibriConfig


log = logging.getLogger(__name__)


class LesHouchesEnvironment(Environment):
    pass


class LesHouchesConfig(colibriConfig):
    """
    GPConfig class Inherits from colibri.config.colibriConfig
    """

    def produce_pdf_model(self, output_path, fitted_flavours):
        """
        Produce the Les Houches model.
        """
        model = LesHouchesPDF(fitted_flavours)
        # dump model to output_path using dill
        # this is mainly needed by scripts/ns_resampler.py
        with open(output_path / "pdf_model.pkl", "wb") as file:
            dill.dump(model, file)
        return model
