#!/bin/bash

conda init bash
conda activate test
conda install nnpdf -y
conda install mpi4py -y
python -m pip install --upgrade pip
python -m pip install flake8 pytest ultranest jax optax flax flit
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi