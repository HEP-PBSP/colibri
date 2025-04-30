.. _lh-closure-test:

============
Closure Test
============


.. _enable-executable:

Step 1: enable the executable
-----------------------------

The first step is to enable the executable for this tutorial. 

In your ``colibri-dev`` conda environment, go to the ``les_houches_example`` directory, found in:

.. code-block:: bash
    
    cd colibri/examples/les_houches_example 

Then run:

.. code-block:: bash
    
    pip install -e .

This will enable an exexutable called ``les_houches_exe``. 

Step 2: runcard
---------------

In the ``colibri/examples/les_houches_example/runcards`` directory you will find
two runcards. ``lh_fit_model_specific_closure_test.yaml`` runs a 
:ref:`model-specific closure test <lh-model-specific-closure-test>`. For this tutorial,
the relevant runcard is ``lh_fit_closure_test.yaml``, which looks like this:

.. literalinclude:: ../../../../../examples/les_houches_example/runcards/lh_fit_closure_test.yaml
    :language: python

TODO: add some discussion on the runcard

Step 3: producing the fit
-------------------------