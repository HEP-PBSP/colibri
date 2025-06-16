.. _installation:

Installing Colibri on Linux or macOS
=====================================

This section covers installing ``colibri`` in various ways.

1. Development Installation via Conda
--------------------------------------

You can install colibri easily by first cloning the repository and then using the provided ``environment.yml`` file

.. code-block:: bash

    git clone https://github.com/HEP-PBSP/colibri
    cd colibri

from your conda base environment run 

.. code-block:: bash

    conda env create -f environment.yml

This will create a ``colibri-dev`` environment installed in development mode.
If you want to use a different environment name you can run:

.. code-block:: bash

    conda env create -n myenv -f environment.yml


2. Installing with pip
-----------------------

If you don't want to clone the repository and don't need to work in development mode you can follow the installation instructions below.

.. note::
   Most of the ``colibri`` dependencies are available in the `PyPi repository <https://pypi.org/>`_, however non-python codes such as LHAPDF and pandoc won't be installed automatically and neeed to be manually installed in the environment. Because of this we recommend to use a conda environment.

Create a conda environment from your base environment, for instance

.. code-block:: bash

    conda create -n colibri-dev python>=3.11

In this new environment install the following conda packages

.. code-block:: bash

    conda install mpich lhapdf pandoc mpi4py ultranest pip

After having completed this you can simply install the rest of the dependencies with ``pip``:

.. code-block:: bash

    python -m pip install git+https://github.com/HEP-PBSP/colibri.git

to verify that the installation went trough

.. code-block:: bash

    python -c "import colibri; print(colibri.__version__)"
    colibri --help


3. GPU (CUDA) JAX Support
--------------------------

The installation instructions shown above will install jax in cpu mode. It is however possible to run
colibri fits using gpu cuda support too.
To do so, after installing the package following one of the methods shown above, if you are on a linux
machine you can install jax in cuda mode by running

.. code-block:: bash

    pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

