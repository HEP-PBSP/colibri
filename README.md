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
- [Contributing](#contributing)
- [License](#license)

---

## Documentation

Documentation regarding the usage and installation is available at <https://hep-pbsp.github.io/colibri/>

## Features
- Perform PDF fits using flexible parametrisations
- Support for JAX-based computations (CPU by default)
- Optional GPU (CUDA) support for accelerated JAX operations
- Command-line scripts for common workflows (`colibri`, `evolve_fit`, etc.)
- Integration with external PDF model repositories

---

## Installation

This section covers installing `colibri` in various ways.

### 1. Development Installation via Conda

You can install colibri easily by first cloning the repository and then using the provided `environment.yml` file

```bash
git clone https://github.com/HEP-PBSP/colibri
cd colibri
```

from your conda base environment run 

```bash
conda env create -f environment.yml

```

This will create a `colibri-dev` environment installed in development mode.
If you want to use a different environment name you can run:

```bash
conda env create -n myenv -f environment.yml
```


### 2. Installing with pip

If you don't want to clone the repository and don't need to work in development mode you can follow the installation instructions below.

> **Note:** 
> Most of the `colibri` dependencies are available in the [PyPi repository](https://pypi.org/), however non-python codes such as LHAPDF and pandoc won’t be installed automatically and need to be manually installed in the environment. Because of this we recommend to use a conda environment.

Create a conda environment from your base environment, for instance

```bash
conda create -n colibri-dev python>=3.11
```

In this new environment install the following conda packages

```bash
conda install mpich lhapdf pandoc mpi4py ultranest pip
```

After having completed this you can simply install the rest of the dependencies with `pip`:

```bash
python -m pip install git+https://github.com/HEP-PBSP/colibri.git
```

Note, this will install the latest development version, if you want to install a specific release you can specify the 
version, for instance for v0.2.0 you can use the following command

```bash
python -m pip install git+https://github.com/HEP-PBSP/colibri.git@v0.2.0

```

to verify that the installation went through

```bash
python -c "import colibri; print(colibri.__version__)"
colibri --help
```


### 3. GPU (CUDA) JAX Support

The installation instructions shown above will install jax in cpu mode. It is however possible to run
colibri fits using gpu cuda support too.
To do so, after installing the package following one of the methods shown above, if you are on a linux
machine you can install jax in cuda mode by running

```bash
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

```

**Note:** 
It is possible to run fits using float32 precision, the only way of doing so currently is to apply a patch to ultranest so that the json.dump is compatible. To do that, follow the instructions:

```bash
git clone git@github.com:LucaMantani/UltraNest.git
cd UltraNest
git switch add-numpy-encoder
pip install .

```


## Usage

Usage examples such as the implementation of a PDF model and running of a PDF fit can be found under Tutorials in the [Documentation](https://hep-pbsp.github.io/colibri/).


## Contributing

We welcome bug reports or feature requests sent to the [issue tracker](https://github.com/HEP-PBSP/colibri/issues). You may use the issue tracker for help and questions as well.

If you would like contribute to the code, we ask you to kindly follow the [NNPDF Contribution Guidelines](https://docs.nnpdf.science/contributing/index.html).

When developing locally, before committing please test your changes by running `pytest` from the root of the repository.


## License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as 
published by the Free Software Foundation.

## Acknowledgments

Artwork by <a href="https://www.instagram.com/qftoons/" target="_blank">@qftoons</a>



