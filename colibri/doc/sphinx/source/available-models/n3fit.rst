.. _n3fit-model:

===========
n3fit Model
===========

**Model Respository:** https://github.com/HEP-PBSP/colibri-n3fit/tree/main/colibri_n3fit .

This model is based on the `n3fit` model used in the NNPDF framework :cite:`NNPDF:2021uiq`,
which is open source and available here: https://github.com/NNPDF/nnpdf .


What is this model for
----------------------

This model is mostly intended to run :ref:`Monte Carlo fits <running_mc_replica>`,
and can be used to compare NNPDF's n3fit model to other models/parametrisations or
fitting methodologies.

Model description
-----------------

This model parametriseses PDFs by the following functional form:

.. math::
    :label: eq:nnpdf-parametrisaton

    f_{i}(x) =  A_i \mathrm{NN}(x)_{j} * x^{1-\alpha_{j}} * (1-x)^{\beta_{j}},

where the PDFs are defined in the evolution basis, as described in Ref. :cite:alp:`NNPDF:2021uiq`,
and :math:`\mathrm{NN}(x)_{j}` is the output of a Neural Network. 

The free parameters of this model are are:

* The preprocessing parameters, :math:`\alpha` and :math:`\beta`, which are sampled for each replica from uniform distributions, as defined by ``FLAV_INFO_NNPDF40`` in ``colibri-n3fit/colibri_n3fit/utils.py``. These values are fixed during training.
* The NN weights, which are found by minimising the :math:`\chi^2` (see :ref:`this section <likelihood>`).

Settings that are specific to this model are described :ref:`below <n3fit-model-settings>`.


How to use this model
---------------------

Before running a fit, you will have to clone the repository:

.. code-block:: bash

    git clone https://github.com/HEP-PBSP/colibri-n3fit/tree/main/colibri_n3fit ,

and install the dependencies and executable:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate example-colibri-n3fit
    pip install -e .

You can then run an fit with the available example runcard:

.. code-block:: bash

    colibri_n3fit colibri_n3fit/runcards/example_pdf_fit_monte_carlo.yaml -r 1

To analyse the results of this fit, you can follow the instructions given in
:ref:`this section <mc_fit_folders>`.


.. _n3fit-model-settings:

Model specific settings
^^^^^^^^^^^^^^^^^^^^^^^

The neural network architecture can be defined in the runcard through the following parameters:

.. code-block::

    nodes: [25,20,8]  # The number of nodes in each layer of the neural network. The output should be set to the number of PDF flavours
    activations: [tanh, tanh, linear]
    nnseed: 945709987       # Seed used for the sampling of preprocessing factors

- ``nodes``: the number of nodes in each hidden layer. The last layer should have a number of nodes equal to the number of PDF flavours being fitted. All flavours in the evolution basis are fitted by default in this model, but you can choose to fit a subset of these with ``flavour_mapping`` if and only if you are running a :ref:`closure test <lh-closure-test>`.
- ``activations``: the activation function to be used in each hidden layer, e.g. ``tanh`` or ``linear``. You can read about other options in the `Keras documentation <https://keras.io/api/layers/activations/>`_ .
- ``nnseed``: The random seed used to sample the preprocessing factors :math:`\alpha` and :math:`\beta` from uniform distributions for each replica. 

You can read more about other relevant settings, such as Monte Carlo or Training settings in the :ref:`section on how to run Monte Carlo fits <running_mc_replica>`.
