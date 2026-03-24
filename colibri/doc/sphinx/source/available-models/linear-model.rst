.. _linear-model:

============
Linear Model
============

This model was presented in Ref. :cite:alp:`Costantini:2025wxp`.

**Model Repository:**  https://github.com/HEP-PBSP/wmin-model .

What is this model for?
-----------------------

This model is especially suitable for :ref:`running bayesian fits <in_running_bayesian>`.
It can be used to:

1. **Bayesian PDF Fits with POD Parametrisation** 
2. **POD Basis Construction** Generate a Proper Orthogonal Decomposition (POD) basis (see Ref. :cite:alp:`Costantini:2025wxp` for details on what this is). 


How to use this model
---------------------

You can find installation instructions in the `model repository <https://github.com/HEP-PBSP/wmin-model>`_. 

Constructing a POD basis
^^^^^^^^^^^^^^^^^^^^^^^^

The following is an example runcard that can be used to construct a POD basis: 

.. code-block:: bash

    meta:
    title: POD basis
    author: Lazy Person
    keywords: ["POD basis", "wmin"]

    # NNPDF Neural Net Architecture settings
    replica_range_settings:
    min_replica: 1  
    max_replica: 1000   # generate replicas numbered 1 to 1000


    impose_sumrule: true
    filter_sr_outliers: false   # whether to filter sum rules outliers

    fitbasis: EVOL

    nodes: [25, 20, 8]

    activations: ["tanh", "tanh", "linear"]

    initializer_name: "glorot_normal"
    layer_type: "dense"


    # Number of components to keep
    Neig: 10

    # theoryid used after SVD to evolve fit
    theoryid: 40_000_000

    actions_:
    - write_pod_basis


This will generate ``max_replica - min_replica`` random initialisations of the n3fit Neural Network,
that will then be reduced to ``Neig`` eigenvectors, which will be the basis elements. It can be run
with the command:

.. code-block:: bash

    wmin runcard.yaml

where ``wmin`` is the model-specific executable.

This basis should then be :ref:`evolved <evolution_script>`, and the basis elements then need to be
shifted by running:

.. code-block:: bash

    python shift_lhapdf_members.py evolved_directory/postfit/evolved_directory

where the ``shift_lhapdf_members.py`` script can be found in the directory ``wmin-model/wmin/runcards``
and ``evolved_directory`` is the fit or POD basis directory that should have previously been evolved.

You can then follow Colibri's :ref:`analytic <running_analytic>` and :ref:`bayesian <in_running_bayesian>`
workflows to run fits.