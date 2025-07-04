.. _lh-closure-test:

======================
Running a Closure Test
======================

In this tutorial, we will demonstrate how to perform a closure test using Colibri. 
We will do so by performing a closure test with the Les Houches parametrisation
model, which was implemented in :ref:`this tutorial <in_les_houches>`.

.. _enable-executable:

Step 1: enable the executable
-----------------------------

The first step is to enable the executable for this model. 

In your ``colibri-dev`` conda environment, go to the ``les_houches_example`` 
directory, found in:

.. code-block:: bash
    
    cd colibri/examples/les_houches_example 

Then run:

.. code-block:: bash
    
    pip install -e .

This will enable an executable called ``les_houches_exe``. 

Step 2: runcard
---------------

In the ``colibri/examples/les_houches_example/runcards`` directory you will find
an example runcard called ``lh_fit_closure_test.yaml``, which looks like this:

.. literalinclude:: ../../../../../examples/les_houches_example/runcards/lh_fit_closure_test.yaml
    :language: python

Note that we will be using the PDF grid ``LH_PARAM_20250519``, which has been 
produced by computing the relevant PDFs for the Les Houches model with the 
best-fit values for the parameters, taken from Ref. :cite:`Alekhin:2005xgg`. 

Note also that, by choosing the action ``run_ultranest_fit``, we are running a 
bayesian fit. You can read more about how to run a bayesian fit in
:ref:`this tutorial <in_running_bayesian>`.

If you instead want to perform a closure test with the monte carlo replica method,
you can find out how to do in :ref:`this tutorial <running_mc_replica>`.


Step 3: producing the fit
-------------------------

To produce the fit, run the following command from the ``colibri/les_houches_example`` 
directory:

.. code-block:: bash

    les_houches_exe runcards/lh_fit_closure_test.yaml

This step will download the PDF grid ``LH_PARAM_20250519``. 

If you don't have it already, it will also download the theory ``40000000``.

After you run the fit the first time, any subsequent fits should be faster.

A directory called ``lh_fit_closure_test``, containing the output of the fit, 
should have been created. You can read more about the fit folders 
:ref:`here <bayes_fit_folders>`.

3.1 Evolving the fit
^^^^^^^^^^^^^^^^^^^^

If you don't already have it, you will need to download the EKO corresponding to 
the theory used in this tutorial :cite:`Candido:2022tld`, :cite:`Candido2022EKO`:

.. code-block:: bash
    
    vp-get EKO 40000000

You can then evolve the fit by running the following command from the 
``les_houches_example`` directory:

.. code-block:: bash

    evolve_fit lh_fit_closure_test

You can read more about evolution in :ref:`this section <evolution_script>`.

3.2 Generating a ``validphys`` report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, you can run:

.. code-block:: bash

    validphys plot_pdf_fits.yaml

to generate a validphys report :cite:`zahari_kassabov_2019_2571601`.

The result
----------

As an example, we show the result of the fit for the gluon PDF. 

.. image:: ../../_static/figures/g_pdf.png
   :width: 600px
   :align: center

The orange line, labelled *LH theory PDF*, shows the gluon PDF computed from 
best-fit values for all parameters. The green curve/section, labelled 
*Les Houches fit 68% c.i. + 1*:math:`\sigma`, shows the result of the closure test
fit with error band.