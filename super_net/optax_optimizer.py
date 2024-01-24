"""
super_net.optax_optimizer.py

Module contains functions for optax gradient descent optimisation.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import optax
from flax.training.early_stopping import EarlyStopping

import logging

log = logging.getLogger(__name__)


def optimizer_provider(
    optimizer: str = "adam", learning_rate: float = 5e-4, optimizer_settings={}
) -> optax._src.base.GradientTransformationExtraArgs:
    """
    Define the optimizer.

    Parameters
    ----------
    optimizer : str
        Name of the optimizer to use.

    learning_rate : float
        Learning rate.

    optimizer_settings : dict
        Dictionary containing the optimizer settings.
    
    Returns
    -------
    optax._src.base.GradientTransformationExtraArgs
        Optax optimizer.

    """
    opt = getattr(optax, optimizer)
    opt_inp = {"learning_rate": learning_rate, **optimizer_settings}

    return opt(**opt_inp)


def early_stopper(
    min_delta=1e-5, patience=20, max_epochs=1000, mc_validation_fraction=0.2
):
    """
    Define the early stopping criteria.
    If mc_validation_fraction is zero then patience is the same as max_epochs.
    """
    if not mc_validation_fraction:
        log.warning(
            "No validation data provided, patience of early stopping set to max_epochs."
        )
        return EarlyStopping(min_delta=min_delta, patience=max_epochs)
    return EarlyStopping(min_delta=min_delta, patience=patience)
