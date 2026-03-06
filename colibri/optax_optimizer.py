"""
colibri.optax_optimizer.py

Module contains functions for optax gradient descent optimisation.

"""

import optax
from flax.training.early_stopping import EarlyStopping

import logging

log = logging.getLogger(__name__)


def optimizer_provider(
    optimizer_settings,
) -> optax._src.base.GradientTransformationExtraArgs:
    """
    Define the optimizer.

    Parameters
    ----------

    optimizer_settings : dict, default = {}
        Dictionary containing the optimizer settings.

    Returns
    -------
    optax._src.base.GradientTransformationExtraArgs
        Optax optimizer.

    """

    optimizer = optimizer_settings["optimizer"]
    log.info(f"Using {optimizer} optimizer.")
    opt = getattr(optax, optimizer)
    optimizer_hyperparams = optimizer_settings.get("optimizer_hyperparams", {})
    log.info(f"Optimizer hyperparameters: {optimizer_hyperparams}.")
    scheduler_cfg = optimizer_settings.get("scheduler")

    if scheduler_cfg is not None:
        scheduler = scheduler_cfg["name"]
        sched_params = scheduler_cfg["params"]
        log.info(f"Using {scheduler} scheduler with params {sched_params}.")
        scheduler_fn = getattr(optax, scheduler)(**sched_params)
        optimizer_hyperparams["learning_rate"] = scheduler_fn

    clipnorm_cfg = optimizer_settings.get("clipnorm")
    if clipnorm_cfg is not None:
        clipnorm = optimizer_settings["clipnorm"]
        log.info(f"Using gradient clipping with norm {clipnorm}.")
        return optax.chain(
            optax.clip_by_global_norm(clipnorm),
            opt(**optimizer_hyperparams),
        )
    else:
        return opt(**optimizer_hyperparams)


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
