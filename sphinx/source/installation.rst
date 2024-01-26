===================================
**Installation**
===================================

Manual installation
-------------------

We recommend installing the code using mamba as follows.

::

    mamba create -n supernet -y && conda activate supernet
    mamba install python=3.10 jax=0.4.13 ml_dtypes optax=0.1.7 flax chex=0.1.83
    mamba install flit -c conda-forge -y
    mamba install lhapdf prompt_toolkit seaborn h5py
    pip install reportengine validobj pineappl "ruamel.yaml<0.18.0" ultranest 

