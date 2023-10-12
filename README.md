# super_net
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