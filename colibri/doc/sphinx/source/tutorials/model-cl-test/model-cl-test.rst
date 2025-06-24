.. _sec_model_cl_test:

===========================
Model-Specific Closure Test 
===========================

Colibri can be used to run a model-specific closure test. Here we show how to 
do so for the Les Houches parametrisation (implemented :ref:`here <lh_model>`).

Step 1: runcard
---------------

An example runcard can be found in ``colibri/examples/les_houches_example/runcards``.
It is meant to run a closure test (see :ref:`in_running_closure`) so, to run a 
model-specific closure test, you should adapt it to include the following:

.. code-block:: bash
    
    # MODEL-SPECIFIC CLOSURE TEST:
    closure_test_pdf: colibri_model
    closure_test_model_settings:
    model: les_houches_example

    # The parameters of the model.
    parameters:
        beta_gluon: 10.9      
        alpha_gluon: 0.356    
        alpha_up: 0.718       
        beta_up: 3.81         
        epsilon_up: -1.56     
        gamma_up: 3.30        
        alpha_down: 1.71      
        beta_down: 10.0       
        epsilon_down: -3.83  
        gamma_down: 4.64      
        norm_sigma: 0.211     
        alpha_sigma: -0.048   
        beta_sigma: 2.20      

And remove or comment out ``closure_test_pdf: LH_PARAM_20250519``.

Note that, unlike for a standard closure test, in this case we need to
specify the initial values for each parameter. For the values above, we 
have taken the best-fit values, taken from Ref. :cite:`Alekhin:2005xgg`. 

Step 2: running the fit
-----------------------

You can produce the fit by running the following command from the 
``colibri/les_houches_example`` directory:

.. code-block:: bash

    les_houches_exe runcards/lh_fit_closure_test.yaml

where the runcard path or name should be changed as needed.

Running the fit will create a directory called ``lh_fit_closure_test``, 
where you will find the output of the fit. You can read more about it in
:ref:`this tutorial <bayes_fit_folders>`.

You can evolve it by following the instructions given in 
:ref:`this tutorial <evolution>`.

