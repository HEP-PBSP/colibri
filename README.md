# super_net
![Tests bagde](https://github.com/HEP-PBSP/colibri/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reportengine app to perform PDF fits in a new parametrisation.


## Super Net Installation
Create a working environment with conda (or mamba):
```
mamba create nnpdf=4.0.6 -n name_environment -y && conda activate name_environment
mamba install -c conda-forge jax optax flax
mamba install flit -c conda-forge -y
cd super_net
flit install --symlink
```

## Installation of Various Models

# Weight Minimization
```
cd wmin
flit install --symlink
```
