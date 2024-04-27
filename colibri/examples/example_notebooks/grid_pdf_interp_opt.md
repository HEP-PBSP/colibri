```python
import jax
import jax.numpy as jnp

from validphys import convolution

from colibri.constants import XGRID

import time
```


```python
inp = {
    "xgrids": {
        "photon": [],
        "\Sigma": [0.01, 0.02, 0.1, 0.2],
        "g": [0.01, 0.02, 0.1, 0.2],
        "V": [0.01, 0.02, 0.1, 0.2],
        "V3": [],
        "V8": [],
        "V15": [],
        "V24": [],
        "V35": [],
        "T3": [],
        "T8": [],
        "T15": [],
        "T24": [],
        "T35": [],
    },
}
```


```python
from validphys.convolution import FK_FLAVOURS

flavour_mapping = [1, 2, 3]

FLAVOURS_ID_MAPPINGS = {
    0: "photon",
    1: "\Sigma",
    2: "g",
    3: "V",
    4: "V3",
    5: "V8",
    6: "V15",
    7: "V24",
    8: "V35",
    9: "T3",
    10: "T8",
    11: "T15",
    12: "T24",
    13: "T35",
}
FLAVOUR_TO_ID_MAPPING = {val: key for (key, val) in FLAVOURS_ID_MAPPINGS.items()}


reduced_xgrids = {FLAVOUR_TO_ID_MAPPING[flav]: val for (flav, val) in inp["xgrids"].items()}

parameters = [
    f"{FK_FLAVOURS[i]}({j})" for i in flavour_mapping for j in reduced_xgrids[i]
]
```


```python
def produce_length_reduced_xgrids(xgrids):
    """The reduced x-grids used in the fit, organised by flavour."""
    lengths = [len(val) for (_, val) in xgrids.items()]
    # Remove all zero-length lists
    lengths = list(filter((0).__ne__, lengths))
    if not all(x == lengths[0] for x in lengths):
        raise ValueError(
            "Cannot currently support reduced x-grids of different lengths."
        )
    return lengths[0]

length_reduced_xgrids = produce_length_reduced_xgrids(inp["xgrids"])
```


```python
def build_xgrid(reduced_xgrids, flavour_mapping):
    out = []
    for fl in flavour_mapping:
        out.append(jnp.array(reduced_xgrids[fl]))
    return jnp.array(out)

fit_xgrid = build_xgrid(reduced_xgrids, flavour_mapping)
```


```python
@jax.jit
def interpolate_1D(y):
    return jnp.interp(jnp.array(XGRID), jnp.array(reduced_xgrids[1]), y)


@jax.jit
def interpolate_2D(y):
    out = []
    for i, xgrid in enumerate(fit_xgrid):
        out.append(jnp.interp(jnp.array(XGRID), xgrid, y[i, :]))
    return jnp.array(out)


@jax.jit
def interpolate_grid_2D(stacked_pdf_grid):
    reshaped_stacked_pdf_grid = stacked_pdf_grid.reshape(
        (
            len(flavour_mapping),
            length_reduced_xgrids,
        ),
    )

    # generate an empty matrix of shape (valipdhys.convolution.NFK, len(colibri.constants.XGRID),)
    input_grid = jnp.zeros(
        (
            convolution.NFK,
            len(XGRID),
        )
    )

    input_grid = input_grid.at[flavour_mapping, :].set(
        interpolate_2D(reshaped_stacked_pdf_grid)
    )

    return input_grid


@jax.jit
def interpolate_grid(stacked_pdf_grid):
    reshaped_stacked_pdf_grid = stacked_pdf_grid.reshape(
        (
            len(flavour_mapping),
            length_reduced_xgrids,
        ),
    )

    # generate an empty matrix of shape (valipdhys.convolution.NFK, len(colibri.constants.XGRID),)
    input_grid = jnp.zeros(
        (
            convolution.NFK,
            len(XGRID),
        )
    )

    input_grid = input_grid.at[flavour_mapping, :].set(
        jnp.apply_along_axis(interpolate_1D, axis=-1, arr=reshaped_stacked_pdf_grid)
    )

    return input_grid


@jax.jit
def interpolate_grid_vec(stacked_pdf_grid):
    reshaped_stacked_pdf_grid = stacked_pdf_grid.reshape(
        (
            stacked_pdf_grid.shape[0],
            len(flavour_mapping),
            length_reduced_xgrids,
        ),
    )

    # generate an empty matrix of shape (:, valipdhys.convolution.NFK, len(colibri.constants.XGRID),)
    input_grid = jnp.zeros(
        (
            stacked_pdf_grid.shape[0],
            convolution.NFK,
            len(XGRID),
        )
    )

    input_grid = input_grid.at[:, flavour_mapping, :].set(
        jnp.apply_along_axis(interpolate_1D, axis=-1, arr=reshaped_stacked_pdf_grid)
    )

    return input_grid


def interpolate_grid_nojit(stacked_pdf_grid):
    reshaped_stacked_pdf_grid = stacked_pdf_grid.reshape(
        (
            len(flavour_mapping),
            length_reduced_xgrids,
        ),
    )

    # generate an empty matrix of shape (valipdhys.convolution.NFK, len(colibri.constants.XGRID),)
    input_grid = jnp.zeros(
        (
            convolution.NFK,
            len(XGRID),
        )
    )

    input_grid = input_grid.at[flavour_mapping, :].set(
        jnp.apply_along_axis(interpolate_1D, axis=-1, arr=reshaped_stacked_pdf_grid)
    )

    return input_grid


def interp_func_OLD(stacked_pdf_grid):
    # reshape stacked_pdf_grid to (len(REDUCED_XGRID), len(flavour_mapping))
    reshaped_stacked_pdf_grid = stacked_pdf_grid.reshape(
        (length_reduced_xgrids, len(flavour_mapping)), order="F"
    )

    # generate an empty matrix of shape (len(colibri.constants.XGRID),valipdhys.convolution.NFK)
    input_grid = jnp.zeros((len(XGRID), convolution.NFK))

    # interpolate columns of reshaped_stacked_pdf_grid
    for i, fl in enumerate(flavour_mapping):
        input_grid = input_grid.at[:, fl].set(
            jnp.interp(
                jnp.array(XGRID),
                jnp.array(reduced_xgrids[fl]),
                reshaped_stacked_pdf_grid[:, i],
            )
        )

    return input_grid.T
```


```python
grids = jax.random.uniform(jax.random.PRNGKey(758493), shape=(10000, 12))
```

# Test speed of different interpolation functions

## Just a loop doing nothing


```python
t0 = time.time()
for grid in grids:
    pass

t1 = time.time()

total = t1 - t0

print("Time for evalutation:", total)
```

    Time for evalutation: 0.03430986404418945


## The new interpolation function 2D


```python
test0 = []
t0 = time.time()
for grid in grids:
    test0.append(interpolate_grid_2D(grid))

t1 = time.time()

total = t1 - t0

print("Time for evalutation:", total)
```

    Time for evalutation: 0.07196879386901855


## New interpolation 1D


```python
test1 = []
t0 = time.time()
for grid in grids:
    test1.append(interpolate_grid(grid))

t1 = time.time()

total = t1 - t0

print("Time for evalutation:", total)
```

    Time for evalutation: 0.07355594635009766


## New interpolation function without JIT compilation


```python
test2 = []
t0 = time.time()
for grid in grids:
    test2.append(interpolate_grid_nojit(grid))

t1 = time.time()

total = t1 - t0

print("Time for evalutation:", total)
```

    Time for evalutation: 9.366427898406982


## The old implementation of the interpolation function


```python
test3 = []
t0 = time.time()
for grid in grids:
    test3.append(interp_func_OLD(grid))

t1 = time.time()

total = t1 - t0

print("Time for evalutation:", total)
```

    Time for evalutation: 29.778298139572144


## The new vectorised function, no for loop


```python
t0 = time.time()

test4 = interpolate_grid_vec(grids)

t1 = time.time()

total = t1 - t0

print("Time for evalutation:", total)
```

    Time for evalutation: 0.007842063903808594


### Check that results are all the same


```python
jnp.all((jnp.array(test1) - jnp.array(test0)) == 0)
```




    Array(True, dtype=bool)




```python
jnp.all((jnp.array(test1) - jnp.array(test2)) == 0)
```




    Array(True, dtype=bool)




```python
jnp.all((jnp.array(test1) - jnp.array(test3)) == 0)
```




    Array(True, dtype=bool)




```python
jnp.all((jnp.array(test1) - test4) == 0)
```




    Array(True, dtype=bool)




```python

```
