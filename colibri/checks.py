from colibri.utils import pdf_models_equal, get_pdf_model
from reportengine.checks import make_argcheck, check


@make_argcheck
def check_pdf_models_equal(prior_settings, pdf_model):
    """
    Decorator that can be added to functions to check that the
    PDF model used as prior (eg when using prior_settings["type"] == "prior_from_gauss_posterior")
    matches the PDF model used in the current fit (pdf_model).
    """
    
    if prior_settings["type"] == "prior_from_gauss_posterior":
        
        prior_fit = prior_settings["prior_fit"]
        prior_pdf_model = get_pdf_model(prior_fit)

        if not pdf_models_equal(prior_pdf_model, pdf_model):
            raise ValueError(f"PDF model {pdf_model} does not match prior settings {prior_pdf_model}")

