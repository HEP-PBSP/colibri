"""
colibri.core.py
This module contains the dataclasses for the core of colibri.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ColibriFitSpec:
    """
    Dataclass for Colibri fit spec.

    Attributes
    ----------
    bayesian_metrics: dict
        Dictionary containing bayesian metrics

    fit_path: str
        Path to the fit.
    """

    bayesian_metrics: dict
    fit_path: str
