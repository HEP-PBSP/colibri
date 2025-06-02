.. _sec_model_cl_test:

===========================
Model-Specific Closure Test 
===========================

Colibri can be used to run a model-specific closure test. Here we show how to 
do so for the Les Houches parametrisation (implemented :ref:`here <lh_model>`).

Step 1: runcard
---------------

An example runcard can be found in ``colibri/examples/les_houches_example/runcards``.
You can adapt it to include the following: 

.. code-block:: bash
    
    # MODEL-SPECIFIC CLOSURE TEST:
    closure_test_pdf: colibri_model
    closure_test_model_settings:
    model: les_houches_example
    fitted_flavours: [\Sigma, g, V, V3]

Note that ``closure_test_pdf: LH_PARAM_20250429`` is not needed any more.

Step 2: running the fit
-----------------------

You can produce the fit by running the following command from the ``colibri/les_houches_example``
directory:

.. code-block:: bash

    les_houches_exe runcards/lh_fit_closure_test.yaml

where the runcard path or name should be changed as needed.

Running the fit will create a directory called ``lh_fit_closure_test``, where you will find
the output of the fit.

You can evolve it by following the instructions given in :ref:`evolution`.