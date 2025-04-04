.. _installation:

Installing Colibri on Linux or macOS
=====================================

Option 1: Using conda
---------------------

From your base conda environment run:

.. code-block:: bash

   conda env create -f environment.yml

This will create a conda environment called `colibri-dev` that has a `colibri` executable and all the needed dependencies. To use a different environment name, one should do:

.. code-block:: bash

   conda env create -n myenv -f environment.yml

Option 2: Using pip
-------------------

Create a working environment with conda (or mamba):

.. code-block:: bash

   conda create -n name_environment -y && conda activate name_environment
   conda install mpich mpi4py lhapdf pandoc
   cd colibri
   pip install -e .

Option 3: Using float32 with Ultranest
--------------------------------------
If using float32 is of interest, one needs to apply a patch to ultranest so that the json.dump is compatible. To do that, follow the instructions:

.. code-block:: bash

   git clone git@github.com:LucaMantani/UltraNest.git
   cd UltraNest
   git switch add-numpy-encoder
   pip install .

**Installation of Various Models**  
The various PDF models, such as `wmin-model` and `gp-model`, should be installed from the respective repositories.



