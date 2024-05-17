"""
wmin.config.py

Config module of wmin

"""

import dill
from validphys.core import PDF
from wmin.model import WMinPDF

from colibri.config import Environment, colibriConfig


class Environment(Environment):
    pass


class WminConfig(colibriConfig):
    """
    WminConfig class Inherits from colibri.config.colibriConfig
    """

    def parse_prior_settings(self, settings):

        if "type" not in settings.keys():
            raise ValueError("Missing key type for prior_settings")

        # Currently, only prior is uniform with max/min val
        if settings["type"] == "uniform_parameter_prior":
            # Check if max and min vals are defined, if not set them to defaults
            # of -1 and 1.
            if "min_val" not in settings.keys():
                settings["min_val"] = -1
            if "max_val" not in settings.keys():
                settings["max_val"] = 1

        return settings

    def produce_pdf_model(self, wmin_settings, output_path):
        """
        Weight minimization grid is in the evolution basis.
        The following parametrization is used:

        f_{j,wm} = f_j + sum_i(w_i * (f_i - f_j))

        this has the advantage of automatically satisfying the sum rules.

        Notes:
            - the central replica of the wminpdfset is always included in the
            wmin parametrization
        """

        model = WMinPDF(PDF(wmin_settings["wminpdfset"]), wmin_settings["n_basis"])

        # dump model to output_path using dill
        # this is mainly needed by scripts/ns_resampler.py
        with open(output_path / "pdf_model.pkl", "wb") as file:
            dill.dump(model, file)
        return model
