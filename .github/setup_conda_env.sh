#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh && conda activate test 2>/dev/null
conda install nnpdf -y
conda install mpi4py -y
python -m pip install --upgrade pip
python -m pip install flake8 pytest ultranest jax optax flax flit
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi