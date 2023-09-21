# super_net
A reportengine app to perform PDF fits in a new parametrisation.

## Installation

```
conda create -n super_net -y && conda activate super_net
conda install mamba python=3.9.2 -c conda-forge -y
mamba install nnpdf=4.0.6 -c https://packages.nnpdf.science/conda -y
mamba install flit -c conda-forge -y
cd super_net
flit install --symlink
```
