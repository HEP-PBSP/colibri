"""
colibri.core.py

This module contains the dataclasses for the core of colibri.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional

"""
Default values for the Colibri fits.
"""
DEFAULT_N_POSTERIOR_SAMPLES = 100
DEFAULT_SAMPLING_SEED = 1
DEFAULT_FULL_SAMPLE_SIZE = 1000
DEFAULT_OPTIMAL_PRIOR = False


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
class ColibriPriorSettings:
    """
    Dataclass containing the specs for the prior of a Colibri fit.

    Attributes
    ----------
    prior_distribution: str
        The prior distribution to be used.

    max_val: float
        The maximum value of the prior.

    min_val: float
        The minimum value of the prior.

    prior_fit: str
        The prior fit to be used.
    """

    prior_distribution: str = None
    max_val: float = None
    min_val: float = None
    prior_fit: str = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class NestedSamplingSettings:
    """
    Dataclass containing the specs for the nested sampling of a Colibri fit.

    Attributes
    ----------
    n_posterior_samples: int
        Number of posterior samples.
    posterior_resampling_seed: int
        Seed for the posterior resampling.
    ReactiveNS_settings: dict
        Settings for the Reactive Nested Sampling.
    Run_settings: dict
        Settings for the run.
    SliceSampler_settings: dict
        Settings for the slice sampler.
    ultranest_seed: int
        Seed for the ultranest.
    sampler_plot: bool
        Whether to plot and save the posterior.
    popstepsampler: bool
        Whether to use the popstepsampler.
    """

    ReactiveNS_settings: dict = field(default_factory=dict)
    Run_settings: dict = field(default_factory=dict)
    n_posterior_samples: int = DEFAULT_N_POSTERIOR_SAMPLES
    posterior_resampling_seed: int = DEFAULT_SAMPLING_SEED
    SliceSampler_settings: dict = field(default_factory=dict)
    ultranest_seed: int = DEFAULT_SAMPLING_SEED
    sampler_plot: bool = True
    popstepsampler: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AnalyticFitSettings:
    """
    Dataclass containing the specs for the analytic fit of a Colibri fit.

    Attributes
    ----------
    n_posterior_samples: int
        Number of posterior samples.
    sampling_seed: int
        Seed for the sampling.
    full_sample_size: int
        Size of the full sample.
    optimal_prior: bool
        Whether to use the optimal prior.
    """

    n_posterior_samples: int = DEFAULT_N_POSTERIOR_SAMPLES
    sampling_seed: int = DEFAULT_SAMPLING_SEED
    full_sample_size: int = DEFAULT_FULL_SAMPLE_SIZE
    optimal_prior: bool = DEFAULT_OPTIMAL_PRIOR

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ColibriSpecs:
    """
    Dataclass containing the settings of a Colibri fit.

    Attributes
    ----------
    loss_function_specs: ColibriLossFunctionSpecs
        The specs for the loss function of the fit.

    prior_settings: ColibriPriorSettings
        The specs for the prior of the fit.

    ns_settings: ColibriNestedSamplingSpecs
        The specs for the nested sampling of the fit.

    analytic_settings: ColibriAnalyticFitSpecs
        The specs for the analytic fit of the fit.
    """

    loss_function_specs: Optional[ColibriLossFunctionSpecs]
    prior_settings: Optional[ColibriPriorSettings]
    ns_settings: Optional[NestedSamplingSettings]
    analytic_settings: Optional[AnalyticFitSettings]


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
