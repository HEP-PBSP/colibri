"""
Module that contains checks for the colibri package.
See validphys/checks.py and reportengine/checks.py for more information / examples.
"""

import yaml
from reportengine.checks import make_argcheck
import jax.numpy as jnp
import jax
from colibri.theory_predictions import make_pred_data, fast_kernel_arrays

from colibri.utils import get_fit_path, get_pdf_model, pdf_models_equal


@make_argcheck
def check_pdf_models_equal(prior_settings, pdf_model, theoryid):
    """
    Decorator that can be added to functions to check that the
    PDF model used as prior (eg when using prior_settings["type"] == "prior_from_gauss_posterior")
    matches the PDF model used in the current fit (pdf_model).
    """

    if prior_settings.prior_distribution == "prior_from_gauss_posterior":

        prior_fit = prior_settings.prior_distribution_specs["prior_fit"]
        prior_pdf_model = get_pdf_model(prior_fit)

        if not pdf_models_equal(prior_pdf_model, pdf_model):
            raise ValueError(
                f"PDF model {pdf_model} does not match prior settings {prior_pdf_model}"
            )

        # load filter.yml runcard of the prior fit
        with open(get_fit_path(prior_fit) / "filter.yml", "r") as file:
            prior_filter = yaml.safe_load(file)

        # check that theory id used in prior fit is the same as the one used in the current fit
        if prior_filter["theoryid"] != theoryid.id:
            raise ValueError(
                f"Theory id {theoryid} does not match theory id of prior {prior_filter['theoryid']}"
            )


@make_argcheck
def check_pdf_model_is_linear(pdf_model, FIT_XGRID, data):
    """
    Decorator that can be added to functions to check that the
    PDF model is linear.
    """

    pred_data = make_pred_data(data, FIT_XGRID)
    fk = fast_kernel_arrays(data, FIT_XGRID)

    parameters = pdf_model.param_names
    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=pred_data)
    intercept = pred_and_pdf(jnp.zeros(len(parameters)), fk)[0]

    # Run the check for 10 random points in the parameter space
    for i in range(10):
        key = jax.random.PRNGKey(i)
        key1, key2 = jax.random.split(key)
        # generate two random points in the parameter space
        x1 = jax.random.uniform(key1, (len(parameters),))
        x2 = jax.random.uniform(key2, (len(parameters),))

        # Test additivity
        add_check = jnp.isclose(
            pred_and_pdf(x1, fk)[0] + pred_and_pdf(x2, fk)[0],
            pred_and_pdf(x1 + x2, fk)[0] + intercept,
        )

        # Test homogeneity
        c = jax.random.uniform(key, (1,))

        homogeneity_check = jnp.isclose(
            c * (pred_and_pdf(x1, fk)[0] - intercept),
            pred_and_pdf(c * x1, fk)[0] - intercept,
        )

        if not add_check.all() or not homogeneity_check.all():
            raise ValueError(
                f"PDF model is not linear or predictions are not linear in the PDFs (e.g. hadronic data is included)."
            )
