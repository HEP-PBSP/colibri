.. _lh-closure-test:

======================
Running a Closure Test
======================

In this tutorial, we will demonstrate how to perform a closure test using Colibri. 
We will do so by performing a closure test with the Les Houches parametrisation
model, which was implemented in :ref:`this tutorial <in_les_houches>`.

Step 1: runcard
---------------

In the ``colibri/examples/les_houches_example/runcards`` directory you will find
an example runcard called ``lh_fit_closure_test.yaml``, which looks like this:

.. literalinclude:: ../../../../../examples/les_houches_example/runcards/lh_fit_closure_test.yaml
    :language: python

:underline:`Points to note:`

* **Underlying law:** We will be using the PDF grid ``LH_PARAM_20250519``, which has been produced by computing the relevant PDFs for the Les Houches model with the best-fit values for the parameters, taken from Ref. :cite:alp:`Alekhin:2005xgg`. 

* **Fitting method:** By choosing the action ``run_ultranest_fit``, we are running a Bayesian fit. (You can read more about how to run a Bayesian fit in :ref:`this tutorial <in_running_bayesian>`.) If you instead want to perform a closure test with the monte carlo replica method, you can find out how to do in :ref:`this tutorial <running_mc_replica>`.

* **Closure test level:** To run a Level 1 closure test with this runcard, you can simply change ``closure_test_level: 0`` to ``1``. To run a model-specific closure, test, see :ref:`this section <model_cl_test>`.

* **Flavour mapping:** This should only be used in closre tests. It masks other flavours, such that closure tests can be run on desired flavours only. 

Step 2: producing the fit
-------------------------

If you have :ref:`enabled the executable for this model <enable-executable>`, you are ready to
run a fit by following command from the ``colibri/les_houches_example`` directory:

.. code-block:: bash

    les_houches_exe runcards/lh_fit_closure_test.yaml

A directory called ``lh_fit_closure_test``, containing the output of the fit, 
should have been created. You can read more about the fit folders 
:ref:`here <bayes_fit_folders>`.

You are now ready to evovle your fit, which you can learn more about in
:ref:`this section <evolution_script>`.

