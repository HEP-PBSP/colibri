import optax
from flax.training.early_stopping import EarlyStopping


def optimizer_provider(
    optimizer: str = "adam", learning_rate: float = 5e-4
) -> optax._src.base.GradientTransformationExtraArgs:
    """ """
    opt = getattr(optax, optimizer)
    return opt(learning_rate=learning_rate)


def early_stopper(min_delta=1e-5, patience=20):
    """ """
    return EarlyStopping(min_delta=min_delta, patience=patience)
