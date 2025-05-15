"""
colibri.core.py

Core module of colibri, containing the main (data) classes for the framework.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorSettings:
    """
    Dataclass containing the settings for the prior transform.

    Attributes
    ----------
    prior_distribution: str
        The type of prior transform.

    prior_distribution_specs: dict
        The settings for the prior distribution.
        Examples: if prior_distribution is "uniform_parameter_prior",
        prior_distribution_specs could be {"max_val": 1.0, "min_val": -1.0}
    """

    prior_distribution: str
    prior_distribution_specs: dict


@dataclass(frozen=True)
class IntegrabilitySettings:
    """
    Dataclass containing the settings for the Integrability constraints
    to be imposed during a fit.

    Attributes
    ----------
    integrability: bool
        Whether to impose integrability constraints.

    integrability_specs: dict
        The settings for the integrability constraints.

    Example
    -------
    integrability_settings:
        integrability: True
        integrability_specs:
            evolution_flavours: [V, V3, V8, T3, T8]
            lambda_integrability: 1000
    """

    integrability: bool
    integrability_specs: dict
