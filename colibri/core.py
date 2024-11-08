"""
TODO
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ColibriTheorySpecs:
    """
    Dataclass containing the theory settings of a Colibri fit.
    The colibri theory is specified by
        - theoryid: the name of the theory
        - cuts: which cuts to be applied to the data

    Note: currently in colibri cuts only supports: `use_cuts: internal`
    """
    theoryid: str    
    use_cuts: str


@dataclass(frozen=True)
class ColibriLossFunctionSpecs:
    """
    Dataclass containing the specs for loss function of a Colibri fit.

    Attributes
    ----------
    use_fit_t0: bool
        whether to perform the fit using the t0-prescription
        This can be needed because of d'Agostini Bias correction (0912.2276)

    t0pdfset: str
        the pdfset to be used for the t0-prescription
    """
    use_fit_t0: bool
    t0pdfset: str


@dataclass(frozen=True)
class ColibriPriorSpecs:
    """
    Dataclass containing the specs for the prior of a Colibri fit.

    Attributes
    ----------
    prior_specs: dict
        Dictionary containing the settings of the prior.
    """
    prior_settings: dict


@dataclass(frozen=True)
class ColibriSpecs:
    """
    Dataclass containing the settings of a Colibri fit.

    Attributes
    ----------
    colibri_specs: dict
        Dictionary containing the settings of the Colibri fit.
    """
    theory_specs: ColibriTheory
    loss_function_specs: ColibriLossFunctionSpecs
    prior_specs: ColibriPriorSpecs


@dataclass(frozen=True)
class ColibriFit:
    """
    Dataclass containing the results and specs of a Colibri fit.

    Attributes
    ----------
    colibri_specs: dataclass

    fit_path: str
        Path to the fit.
    
    """

    colibri_specs: ColibriSpecs
    fit_path: str
    