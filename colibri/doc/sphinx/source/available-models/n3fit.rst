.. _n3fit-model:

===========
n3fit Model
===========

**Model Respository:** https://github.com/HEP-PBSP/colibri-n3fit/tree/main/colibri_n3fit

This model is based on the `n3fit` model used in the NNPDF framework, which is open source and available here: https://github.com/NNPDF/nnpdf

This PDF model is parameterised by the following functional form:

.. math::
    f_{i}(x) =  A_i NN(x)_{j} * x^{1-alpha_{j}} * (1-x)^{beta_{j}}

where the PDFs are defined in the evolution basis as described in :cite:NNPDF:2021uiq. The preprocessing parameters :math:`\alpha` and :math:`\beta`
are sampled for each replica from uniform distributions as defined by `FLAV_INFO_NNPDF40` in `utils.py`, these values are fixed during training.
The neural network architecture can be defined in the runcard through the following parameters:

- `nodes`: the number of nodes in each hidden layer. The last layer should have a number of nodes equal to the number of PDF flavours being fitted.
- `activations`: the activation function to be used in each hidden layer, e.g. `tanh` or `linear`.


How to use this model
---------------------

Clone the repository:

.. code-block:: bash

    git clone https://github.com/HEP-PBSP/colibri-n3fit/tree/main/colibri_n3fit .

Install the dependencies and executable:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate example-colibri-n3fit
    pip install -e .

Run an example fit:

.. code-block:: bash

    colibri_n3fit colibri_n3fit/runcards/example_pdf_fit_monte_carlo.yaml -r 1

To analyse the results of this fit, follow the instructions given in :ref:`this section <mc_fit_folders>`.