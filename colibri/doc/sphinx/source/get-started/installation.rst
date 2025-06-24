.. _installation:

Installing Colibri on Linux or macOS
=====================================

Clone the repository
--------------------
The first step is to clone the repository. You can do this by running the following command in your terminal:

.. code-block:: bash

   git clone https://github.com/HEP-PBSP/colibri

After you have cloned the repository, navigate to the `colibri` directory:

.. code-block:: bash

   cd colibri

Now you can continue with the installation, for which you have two options, as described below.

Option 1: Using conda
---------------------

From your base conda environment run:

.. code-block:: bash 

   conda env create -f environment.yml

This will create a conda environment called ``colibri-dev`` that has a ``colibri`` executable and all the needed dependencies. To use a different environment name, you should do:

.. code-block:: bash

   conda env create -n myenv -f environment.yml

Then you can activate the environment with:

.. code-block:: bash

   conda activate colibri-dev

where you should change ``colibri-dev`` to the name of the environment you created, if you used a different name.

You are ready to start using Colibri! For example, you could head to one of the :ref:`Tutorials <in_tutorials>`.

Option 2: Using pip
-------------------

Create a working environment with conda (or mamba):

.. code-block:: bash

   conda create -n name_environment -y && conda activate name_environment
   conda install mpich mpi4py lhapdf pandoc
   cd colibri
   pip install -e .

Then you can activate the environment with:

.. code-block:: bash

   conda activate name_environment

and you are ready to start using Colibri, for example by following one of the :ref:`Tutorials <in_tutorials>`.

Option 3: Using float32 with Ultranest
--------------------------------------
If using float32 is of interest, one needs to apply a patch to ultranest so that the json.dump is compatible. To do that, run the following commands:

.. code-block:: bash

   git clone git@github.com:LucaMantani/UltraNest.git
   cd UltraNest
   git switch add-numpy-encoder
   pip install .

**Installation of Various Models**  
The various PDF models, such as `wmin-model` and `gp-model`, should be installed from the respective repositories.



