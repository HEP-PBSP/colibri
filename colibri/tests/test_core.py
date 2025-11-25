"""
test_core.py

"""

from dataclasses import FrozenInstanceError
import pytest
import jax.numpy as jnp
from colibri.core import TrainValidationSplit, MCPseudodata


# Helper: assert that two jnp arrays are equal (works for scalars too)
def arrays_equal(a, b) -> bool:
    # jnp.array_equal returns a boolean-like object; ensure Python bool
    return bool(jnp.array_equal(a, b))


def assert_dict_arrays_equal(d1: dict, d2: dict):
    """Assert two dicts have the same keys and that each value is an array equal."""
    assert set(d1.keys()) == set(d2.keys()), "Dict keys differ"
    for k in d1.keys():
        val1 = d1[k]
        val2 = d2[k]
        # If values are arrays, compare elementwise; otherwise exact compare.
        if isinstance(val1, (jnp.ndarray,)):
            assert arrays_equal(val1, val2), f"Array values differ for key {k}"
        else:
            assert val1 == val2, f"Values differ for key {k}"


def test_train_validation_split_to_dict_and_contents():
    train = jnp.array([1.0, 2.0, 3.0])
    val = jnp.array([0.1, 0.2])
    tv = TrainValidationSplit(training=train, validation=val)

    d = tv.to_dict()
    # expected dict produced by asdict
    expected = {"training": train, "validation": val}

    assert isinstance(d, dict)
    assert_dict_arrays_equal(d, expected)

    # ensure original arrays are preserved (identity not required, but contents equal)
    assert arrays_equal(d["training"], train)
    assert arrays_equal(d["validation"], val)


def test_train_validation_split_is_frozen():
    t = TrainValidationSplit(training=jnp.array([0]), validation=jnp.array([1]))
    with pytest.raises(FrozenInstanceError):
        # Attempt to mutate attribute should raise FrozenInstanceError
        t.training = jnp.array([9])


def test_mcpseudodata_to_dict_and_defaults():
    pseudodata = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    train_idx = jnp.array([0, 1])
    val_idx = jnp.array([2])
    mc = MCPseudodata(
        pseudodata=pseudodata,
        training_indices=train_idx,
        validation_indices=val_idx,
        # do not set trval_split to test default separately
    )

    d = mc.to_dict()
    expected = {
        "pseudodata": pseudodata,
        "training_indices": train_idx,
        "validation_indices": val_idx,
        "trval_split": False,
    }

    assert isinstance(d, dict)
    assert_dict_arrays_equal(d, expected)

    # default for trval_split is False on construction when not passed
    mc_default = MCPseudodata(pseudodata, train_idx, val_idx)
    assert mc_default.trval_split is False


def test_mcpseudodata_trval_split_can_be_set_and_is_frozen():
    pseudodata = jnp.array([0.0])
    train_idx = jnp.array([0])
    val_idx = jnp.array([0])
    mc = MCPseudodata(
        pseudodata=pseudodata,
        training_indices=train_idx,
        validation_indices=val_idx,
        trval_split=True,
    )

    # ensure the value was set correctly
    assert mc.trval_split is True

    # attempt to mutate -> FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        mc.trval_split = False
