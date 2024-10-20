"""
colibri.optax_optimizer.py

Module contains functions for optax gradient descent optimisation.

"""

import optax
from flax.training.early_stopping import EarlyStopping

import logging

log = logging.getLogger(__name__)


def optimizer_provider(
    optimizer="adam", optimizer_settings={}
) -> optax._src.base.GradientTransformationExtraArgs:
    """
    Define the optimizer.

    Parameters
    ----------
    optimizer : str, default = "adam"
        Name of the optimizer to use.

    optimizer_settings : dict, default = {}
        Dictionary containing the optimizer settings.

    Returns
    -------
    optax._src.base.GradientTransformationExtraArgs
        Optax optimizer.

    """
    # if optimizer_settings is empty, fill it with the default values
    if not "learning_rate" in optimizer_settings.keys():
        optimizer_settings["learning_rate"] = 5e-4

    opt = getattr(optax, optimizer)

    return opt(**optimizer_settings)


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
