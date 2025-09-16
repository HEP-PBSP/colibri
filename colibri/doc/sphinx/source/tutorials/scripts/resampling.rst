.. _resampling_script:

Resampling script
-----------------

The final step in a Bayesian or analytic fit in Colibri (before evolution)
is to write out posterior samples as ``.exportgrid`` files in the ``replicas\``
folder. These samples will then be evolved into a PDF set by the
:ref:`evolution script <evolution_script>`.

As explained in the :ref:`tutorial on running a Bayesian fit <ultranest_runcard>`,
you can decide how many samples are written out in the runcard. Here we give more
details on the resampling settings, which apply both to Bayesain and analytic fits.


For a Bayesian fit using the analytical inference method, you can set the total
number of posterior draws via the ``analytic_settings`` block. For example:

.. code-block:: yaml

    # Analytic settings
    analytic_settings:
      n_posterior_samples: 100
      full_sample_size: 50000

Likewise, if you instead use the UltraNest nested sampler, you can specify the
same parameter name under ``ultranest_settings``:

.. code-block:: yaml

    # ultranest settings
    ultranest_settings:
      n_posterior_samples: 100
      ...


**Key Parameters**

- ``n_posterior_samples``: 
  The number of individual posterior draws that will each be written out as a separate
  ``.exportgrid`` file in the ``replicas/`` folder. If it is not specified, a default
  of 1000 samples will be written out.

- ``full_sample_size`` *(analytic only)* : 
  The total size of the merged posterior sample, which is saved to
  ``full_posterior_sample.csv`` at the top level of your fit directory.

.. note::
    
    In the case of a fit done using the ``ultranest`` nested sampling sampler, 
    the ``full_sample_size`` defaults to an internal number that might depends on the 
    specific run.


If you want to draw additional replicas (or have a smaller set for a finite-size effects
studies) from the posterior distribution of an already-completed PDF fit, you do **not**
need to re-run the full fit. Instead, use the ``resample_fit`` helper script.

**Usage**

To see all available options, invoke:

.. code-block:: console

    $ resample_fit --help

This will print out a help message that looks like this:


.. code-block:: bash

   usage: resample_fit [-h] [--fitype FITYPE] [--nreplicas NREPLICAS] [--resampling_seed RESAMPLING_SEED]
                       [--resampled_fit_name RESAMPLED_FIT_NAME] [--parametrisation_scale PARAMETRISATION_SCALE]
                       fit_name
   
   Script to resample from Bayesian posterior
   
   positional arguments:
     fit_name              The colibri fit from which to sample.
   
   options:
     -h, --help            show this help message and exit
     --fitype FITYPE, -t FITYPE
                           The type of fit to be resampled. Currently only `ultranest` and `analytic` are supported.
     --nreplicas NREPLICAS, -nrep NREPLICAS
                           The number of samples.
     --resampling_seed RESAMPLING_SEED, -seed RESAMPLING_SEED
                           The random seed to be used to sample from the posterior.
     --resampled_fit_name RESAMPLED_FIT_NAME, -newfit RESAMPLED_FIT_NAME
                           The name of the resampled fit.
     --parametrisation_scale PARAMETRISATION_SCALE, -Q PARAMETRISATION_SCALE
                           The scale at which the PDFs are fitted.

As an example, if we want to resample from the posterior distribution of an analytical fit called ``my_fit``
we can do it as follows:

.. code-block:: bash

   resample_fit my_fit -t analytic -n 100 -seed 1234 -newfit my_resampled_fit

.. note::
    
    In order to resample from the posterior distribution of a fit, you need to be
    in the same environment as the one used to perform the fit. Hence, if you want
    to resample a fit done using the Les Houches model, you need to be in the
    environment where the ``les_houches_exe`` exectuable is available.