"""
TODO
"""

from dataclasses import dataclass
from typing import Optional


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
    prior_settings: dict
        Dictionary containing the settings of the prior.
    """

    prior_settings: dict


@dataclass(frozen=True)
class ColibriNestedSamplingSpecs:
    """
    Dataclass containing the specs for the nested sampling of a Colibri fit.

    Attributes
    ----------
    nested_sampling_specs: dict
        Dictionary containing the settings of the nested sampling.
    """

    ns_settings: dict


@dataclass(frozen=True)
class ColibriAnalyticFitSpecs:
    """
    Dataclass containing the specs for the analytic fit of a Colibri fit.

    Attributes
    ----------
    analytic_fit_specs: dict
        Dictionary containing the settings of the analytic fit.
    """

    analytic_settings: dict


@dataclass(frozen=True)
class ColibriSpecs:
    """
    Dataclass containing the settings of a Colibri fit.

    Attributes
    ----------
    TODO
    """

    loss_function_specs: Optional[ColibriLossFunctionSpecs]
    prior_settings: Optional[ColibriPriorSpecs]
    ns_settings: Optional[ColibriNestedSamplingSpecs]
    analytic_settings: Optional[ColibriAnalyticFitSpecs]


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
