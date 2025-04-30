.. _lh-model-specific-closure-test:

===========================
Model-Specific Closure Test 
===========================


Step 2: runcard
---------------

In the ``colibri/examples/les_houches_example/model_specific_closure_test/runcard``
directory you will find the following runcard:

.. literalinclude:: ../../../../../examples/les_houches_example/runcards/lh_fit_model_specific_closure_test.yaml
    :language: python


TODO: add some discussion on the runcard. 

Step 3: producing the fit
-------------------------

From the ``colibri/les_houches_example/model_specific_closure_test`` directory, run the following command:

.. code-block:: bash
    
    les_houches_exe les_houches_fit.yaml

This step should take less than 3 minutes, unless you need to download the theory ``40000000``, in which case it will
take longer, but only the first time you run it.

After the fit is completed, a directory called ``les_houches_fit``, where you will find the output of the run. 

3.1 Evolving the fit
^^^^^^^^^^^^^^^^^^^^

You can evolve the fit using the command, again from the ``model_specific_closure_test`` directory:

.. code-block:: bash
    
    evolve_fit evolve les_houches_fit

3.2 Generating a ``validphys`` report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, in the ``model_specific_closure_test`` directory, you can find a script called ``plot_pdf.yaml``. This script 
will generate a ``validphys`` report for the fit. You can run it with the following command:

.. code-block:: bash
    
    validphys plot_pdf.yaml

The result
----------

TODO: add a figure to show an example of the output. 

