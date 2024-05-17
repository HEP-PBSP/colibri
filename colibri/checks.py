"""
Module that contains checks for the colibri package.
See validphys/checks.py and reportengine/checks.py for more information / examples.
"""

import yaml
from reportengine.checks import make_argcheck

from colibri.utils import get_fit_path, get_pdf_model, pdf_models_equal


@make_argcheck
def check_pdf_models_equal(prior_settings, pdf_model, theoryid, t0pdfset):
    """
    Decorator that can be added to functions to check that the
    PDF model used as prior (eg when using prior_settings["type"] == "prior_from_gauss_posterior")
    matches the PDF model used in the current fit (pdf_model).
    """

    if prior_settings["type"] == "prior_from_gauss_posterior":

        prior_fit = prior_settings["prior_fit"]
        prior_pdf_model = get_pdf_model(prior_fit)

        if not pdf_models_equal(prior_pdf_model, pdf_model):
            raise ValueError(
                f"PDF model {pdf_model} does not match prior settings {prior_pdf_model}"
            )

        # load filter.yml runcard of the prior fit
        with open(get_fit_path(prior_fit) / "filter.yml", "r") as file:
            prior_filter = yaml.safe_load(file)

        # check that theory id used in prior fit is the same as the one used in the current fit
        if str(prior_filter["theoryid"]) != theoryid.id:
            raise ValueError(
                f"Theory id {theoryid} does not match theory id of prior {prior_filter['theoryid']}"
            )

        # check that t0pdfset used in prior fit is the same as the one used in the current fit
        if prior_filter["t0pdfset"] != t0pdfset.name:
            raise ValueError(
                f"t0pdfset {theoryid.t0pdfset} does not match t0pdfset of prior {prior_filter['t0pdfset']}"
            )
