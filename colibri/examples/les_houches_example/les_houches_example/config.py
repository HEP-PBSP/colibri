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
    LesHouchesConfig class Inherits from colibri.config.colibriConfig
    """

    def produce_pdf_model(self, output_path, dump_model=True):
        """
        Produce the Les Houches model.
        """
        model = LesHouchesPDF()
        # dump model to output_path using dill
        # this is mainly needed by scripts/bayesian_resampler.py
        if dump_model:
            with open(output_path / "pdf_model.pkl", "wb") as file:
                dill.dump(model, file)
        return model
