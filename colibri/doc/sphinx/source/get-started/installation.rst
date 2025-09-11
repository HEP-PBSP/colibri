.. _installation:

Installing Colibri on Linux or macOS
====================================

This section covers installing Colibri in various ways, including via **Conda**,
with ``pip`` and with **GPU (CUDA) JAX Support**.

1. Development Installation via Conda
-------------------------------------

You can install Colibri easily by first cloning the repository and then using the
provided ``environment.yml`` file

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
----------------------

If you don't want to clone the repository and don't need to work in development mode,
you can instead install Colibri with ``pip``.

.. note::
   Most of the Colibri dependencies are available in the `PyPi repository <https://pypi.org/>`_, however non-python codes such as LHAPDF and pandoc won't be installed automatically and need to be manually installed in the environment. Because of this, we recommend the use of a conda environment.

The first step is to create a conda environment from your base environment. For instance:

.. code-block:: bash

    conda create -n colibri-dev python>=3.11

In this new environment, install the following conda packages:

.. code-block:: bash

    conda install mpich lhapdf pandoc mpi4py ultranest pip

After having completed this, you can simply install the rest of the dependencies with ``pip``:

.. code-block:: bash

    python -m pip install git+https://github.com/HEP-PBSP/colibri.git

Note that this will install the latest development version. If you want to install a specific
release you should specify the version. For instance, for v0.2.0 you can use the following
command:

.. code-block:: bash

    python -m pip install git+https://github.com/HEP-PBSP/colibri.git@v0.2.0

To verify that the installation went through, run:

.. code-block:: bash

    python -c "import colibri; print(colibri.__version__)"
    colibri --help


3. GPU (CUDA) JAX Support
-------------------------

The installation instructions shown above will install jax in *cpu* mode. It is also
possible, however, to run Colibri fits using *gpu cuda support*. You will first have
to install the package following one of the methods described above. Then, if you
are on a linux machine, you can install jax in cuda mode by running:

.. code-block:: bash

    pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


Note that it is possible to run fits using *float32* precision. The only way of doing
so currently is to apply a patch to UltraNest so that ``json.dump`` is compatible.
To do this, run the following commands:

.. code-block:: bash

    git clone git@github.com:LucaMantani/UltraNest.git
    cd UltraNest
    git switch add-numpy-encoder
    pip install .


