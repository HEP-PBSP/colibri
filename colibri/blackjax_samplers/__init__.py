"""Registry of BlackJAX sampler implementations."""

from importlib import import_module

SAMPLERS = {
    "nested_sampling": "colibri.blackjax_samplers.nested_sampling",
    "nuts": "colibri.blackjax_samplers.nuts",
}


def get_sampler(algorithm="nested_sampling"):
    """Import and return the selected sampler module."""
    if not isinstance(algorithm, str) or algorithm not in SAMPLERS:
        choices = ", ".join(SAMPLERS)
        raise ValueError(
            f"Unknown BlackJAX algorithm {algorithm!r}. " f"Choose from: {choices}."
        )

    return import_module(SAMPLERS[algorithm])
