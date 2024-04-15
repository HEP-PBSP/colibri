# colibri
![Tests bagde](https://github.com/HEP-PBSP/colibri/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reportengine app to perform PDF fits in a new parametrisation.


## colibri Installation
Create a working environment with conda (or mamba):
```
conda create -n name_environment -y && conda activate name_environment
conda install conda install mpich mpi4py lhapdf pandoc
cd colibri
pip install -e .
```

## Installation of Various Models

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
