"""Registry of BlackJAX sampler implementations."""

SAMPLERS = ("nested_sampling", "nuts")


def get_sampler(algorithm="nested_sampling"):
    """Import and return the selected sampler module."""
    if not isinstance(algorithm, str) or algorithm not in SAMPLERS:
        choices = ", ".join(SAMPLERS)
        raise ValueError(
            f"Unknown BlackJAX algorithm {algorithm!r}. Choose from: {choices}."
        )

    if algorithm == "nested_sampling":
        from . import nested_sampling

        return nested_sampling

    from . import nuts

    return nuts
