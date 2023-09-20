import optax
from flax.training.early_stopping import EarlyStopping


def optimizer_provider(
    optimizer: str = "adam", learning_rate: float = 5e-4, weight_decay=2
) -> optax._src.base.GradientTransformationExtraArgs:
    """ """
    opt = getattr(optax, optimizer)
    kwargs = {"learning_rate": learning_rate}

    # Check if the optimizer has the weight_decay argument
    if "weight_decay" in opt.__code__.co_varnames:
        kwargs["weight_decay"] = weight_decay

    return opt(**kwargs)
    


def early_stopper(min_delta=1e-5, patience=20):
    """ """
    return EarlyStopping(min_delta=min_delta, patience=patience)
