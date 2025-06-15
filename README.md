<div align="center">
<img src="./logo_colibri.png" alt="colibri logo" width="40%">
</div>

<div style="text-align: right; font-size: 10px;">
    Artwork by <a href="https://www.instagram.com/qftoons/" target="_blank">@qftoons</a>
</div>

# colibri
![Tests badge](https://github.com/HEP-PBSP/colibri/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/HEP-PBSP/colibri/graph/badge.svg?token=BQ01FTYGZO)](https://codecov.io/gh/HEP-PBSP/colibri)

A reportengine app to perform PDF fits using arbitrary parametrisations.

---

## Table of Contents
- [Documentation](#documentation)
- [Features](#features)
- [Installation](#installation)
  - [1. Core Installation via Conda](#1-core-installation-via-conda)
  - [2. Core Installation via pip](#2-core-installation-via-pip)
  - [3. GPU (CUDA) JAX Support](#3-gpu-cuda-jax-support)
- [Usage](#usage)
- [Installation of Various Models](#installation-of-various-models)
- [Development & Testing](#development--testing)
- [Contributing](#contributing)
- [License](#license)

---

## Documentation

TODO

## Features
- Perform PDF fits using flexible parametrisations
- Support for JAX-based computations (CPU by default)
- Optional GPU (CUDA) support for accelerated JAX operations
- Command-line scripts for common workflows (`colibri`, `evolve_fit`, etc.)
- Integration with external PDF model repositories

---

## Installation

This section covers installing `colibri` in various ways. By default, `pip install hep-colibri` installs JAX CPU build and all core dependencies. For GPU support, see the separate section below.

### 1. Core Installation via Conda

**Option A: Using provided environment.yml**

```bash
conda env create -f environment.yml

```

To use a different environment name:

```bash
conda env create -n myenv -f environment.yml
```

**Option B: Manual Conda environment**

```bash
conda create -n colibri-env -y python=3.10
conda activate colibri-env
conda install mpich mpi4py lhapdf pandoc
pip install hep-colibri
```

### 2. Core Installation via pip

From your python environment run

```bash
pip install --upgrade pip
pip install hep-colibri
```

to verify that the installation went trough

```bash
python -c "import colibri; print(colibri.__version__)"
colibri --help
```


### 3. GPU (CUDA) JAX Support

#### TODO


## Usage

#### TODO


## Installation of various models


#### TODO


## Development & Testing

The advised way of installing to work in development mode is: TODO

```bash
git clone https://github.com/HEP-PBSP/colibri.git
cd colibri
```


## License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as 
published by the Free Software Foundation

## Acknowledgments

Artwork by <a href="https://www.instagram.com/qftoons/" target="_blank">@qftoons</a>



