.. _model_cl_test:

===========================
Model-Specific Closure Test 
===========================

Colibri can be used to run model-specific closure tests. Here we show how to 
do so for the Les Houches parametrisation (implemented :ref:`here <lh_model>`).

Step 1: runcard
---------------

The runcard ``runcards/lh_closure_test_model.yaml`` in the ``examples/les_houches_example``
directory runs such a closure test. It is the same as the runcard presented
:ref:`here <lh-closure-test>`, except that ``closure_test_pdf: LH_PARAM_20250519``
is replaced by the following block:

.. literalinclude:: ../../../../../examples/les_houches_example/runcards/lh_closure_test_model.yaml
    :language: yaml
    :start-at: closure_test_pdf: colibri_model
    :end-before: ####

Within the ``closure_test_model_settings``, you need to specify any of the hyperparameters
relevant to build the model instance. The example model shown here doesn't need further
settings, but other models may need additional hyperparameters to tune.

Note that, unlike for a standard closure test, in this case we need to
specify the initial values for each parameter of the PDF model. For the values above, we 
have taken the best-fit values, taken from Ref. :cite:alp:`Alekhin:2005xgg`. 

Step 2: running the fit
-----------------------

You can produce the fit by running the following command from the 
``examples/les_houches_example`` directory:

.. code-block:: bash

    les_houches_exe runcards/lh_closure_test_model.yaml

Running the fit will create a directory called ``lh_closure_test_model``, 
where you will find the output of the fit. You can read more about it in
:ref:`this tutorial <bayes_fit_folders>`.

You can evolve it by following the instructions given in 
:ref:`this tutorial <evolution_script>`.