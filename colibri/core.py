"""
colibri.core.py

Core module of colibri, containing the main (data) classes for the framework.
"""

from dataclasses import dataclass, asdict
import jax.numpy as jnp
from typing import Callable, Any, Dict


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
        integrability_specs: {'evolution_flavours': [V, V3, V8, T3, T8], 'lambda_integrability': 1000}
    """

    integrability: bool
    integrability_specs: dict


@dataclass(frozen=True)
class BayesianFit:
    """
    Dataclass containing the results and specs of a Bayesian fit.

    Attributes
    ----------
    param_names: list
        List of the names of the parameters.
    resampled_posterior: jnp.array
        Array containing the resampled posterior samples.
    full_posterior_samples: jnp.array
        Array containing the full posterior samples.
    bayes_complexity: float
        The Bayesian complexity of the model.
    avg_chi2: float
        The average chi2 of the model.
    min_chi2: float
        The minimum chi2 of the model.
    logz: float
        The log evidence of the model.
    """

    param_names: list
    resampled_posterior: jnp.array
    full_posterior_samples: jnp.array
    bayesian_metrics: dict


@dataclass(frozen=True)
class AnalyticFit(BayesianFit):
    """
    Dataclass containing the results and specs of an analytic fit.

    Attributes
    ----------
    analytic_specs: dict
        Dictionary containing the settings of the analytic fit.
    resampled_posterior: jnp.array
        Array containing the resampled posterior samples.
    """

    analytic_specs: dict


@dataclass(frozen=True)
class UltranestFit(BayesianFit):
    """
    Dataclass containing the results and specs of an Ultranest fit.

    Attributes
    ----------
    ultranest_specs: dict
        Dictionary containing the settings of the Ultranest fit.
    ultranest_result: dict
        result from ultranest, can be used eg for corner plots
    """

    ultranest_specs: dict
    ultranest_result: dict


@dataclass(frozen=True)
class HessianFit:
    """
    Dataclass containing the results and specs of a Hessian fit.

    Attributes
    ----------
    hessian_specs: dict
        Dictionary containing the settings of the Hessian fit.
    min_chi2: jnp.ndarray
        Array containing the minimum chi-squared value.
    training_loss: jnp.ndarray
        Array containing the training loss values during the fit.
    optimized_parameters: jnp.ndarray
        Array containing the optimized parameters in the minimum of the chi2.
    hessian: jnp.ndarray
        Array containing the Hessian matrix at the minimum of the chi2.
    cov_params: jnp.ndarray
        Array containing the covariance matrix of the parameters.
        Computed as the inverse of the Hessian matrix.
    resampled_posterior: jnp.ndarray
        Array containing the samples of the parameters drawn from a multivariate
        normal distribution with mean at the optimized parameters and covariance
        given by cov_params.
    """

    hessian_specs: dict
    min_chi2: jnp.ndarray
    training_loss: jnp.ndarray
    optimized_parameters: jnp.ndarray
    hessian: jnp.ndarray
    cov_params: jnp.ndarray
    resampled_posterior: jnp.ndarray
    param_names: list


@dataclass(frozen=True)
class MonteCarloFit:
    """
    Dataclass containing the results and specs of a Monte Carlo fit.

    Attributes
    ----------
    monte_carlo_specs: dict
        Dictionary containing the settings of the Monte Carlo fit.
    training_loss: jnp.array
        Array containing the training loss.
    validation_loss: jnp.array
        Array containing the validation loss.
    optimized_parameters: jnp.array
        Array containing the optimized parameters.
    parameters_by_epoch: jnp.array
        Array containing the parameters of the model recorded during training.
    """

    monte_carlo_specs: dict
    training_loss: jnp.array
    validation_loss: jnp.array
    optimized_parameters: jnp.array
    parameters_by_epoch: jnp.array


@dataclass(frozen=True)
class GradientDescentResult:
    """Result of a generic gradient descent run.

    Attributes
    ----------
    optimized_parameters: jnp.ndarray
        Final optimized parameters.
    training_loss: jnp.array
        Recorded (epoch) training losses (sampled according to record_every).
    validation_loss: jnp.array
        Recorded (epoch) validation losses (sampled according to record_every).
    specs: dict
        Dictionary of settings used for the run (epochs, batch size, etc.).
    parameters_by_epoch: jnp.array
        Array containing the parameters of the model recorded during training.
    """

    optimized_parameters: Any
    training_loss: jnp.array
    validation_loss: jnp.array
    specs: Dict[str, Any]
    parameters_by_epoch: jnp.array


@dataclass(frozen=True)
class MCPseudodata:
    pseudodata: jnp.array
    training_indices: jnp.array
    validation_indices: jnp.array
    trval_split: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DataBatches:
    data_batch_stream_index: Callable
    num_batches: int
    batch_size: int


@dataclass(frozen=True)
class TrainValidationSplit:
    training: jnp.array
    validation: jnp.array

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PosdataTrainValidationSplit(TrainValidationSplit):
    n_training: int
    n_validation: int


@dataclass(frozen=True)
class CentralCovmatIndex:
    central_values: jnp.array
    covmat: jnp.array
    central_values_idx: jnp.array

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CentralInvCovmatIndex:
    central_values: jnp.array
    inv_covmat: jnp.array
    central_values_idx: jnp.array

    def to_dict(self):
        return asdict(self)
