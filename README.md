# colibri
![Tests bagde](https://github.com/HEP-PBSP/colibri/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reportengine app to perform PDF fits in a new parametrisation.


## colibri Installation
- Option 1:
From your base conda environment run:
```
conda env create -f environment.yml
```
this will create a conda environment called `colibri-dev` that has a `colibri` executable and all the needed dependencies. To use a different environment name, one should do
```
conda env create -n myenv -f environment.yml
```

- Option 2:
From you base conda environment run:
```
conda env create -f full_environment.yml
```
this will create a conda environment called `colibri-dev` that has a `colibri` executable, all the needed dependencies as well
as the other subpackages such as `wmin` and `grid_pdf`.

- Option 3:
Create a working environment with conda (or mamba):
```
conda create -n name_environment -y && conda activate name_environment
conda install mpich mpi4py lhapdf pandoc
cd colibri
pip install -e .
```

## Installation of Various Models
In general if you are not using the `full_environment.yml` installation, it is possible to manually install the various models.
# Weight Minimization
```
cd models/grid_pdf
pip install -e .
```

# Weight Minimization
```
cd models/wmin
pip install -e .
```
