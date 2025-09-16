.. _colibri_fit_folders:

Colibri fit folders
-------------------
A Colibri fit folder is the output resulting from a Colibri fit. It is a folder
containing a set of relevant information for the fit.
Currently, we distinguish between two types of fit folders: Bayesian fit folders and 
Monte Carlo replica fit folders.

.. _bayes_fit_folders:

Bayesian fit folders
^^^^^^^^^^^^^^^^^^^^

.. note::

   By “Bayesian fit folder” we mean a folder containing the results of a fit
   performed with a Bayesian sampling method (see :ref:`this section <in_running_bayesian>`
   for details on how to run a Bayesian fit).

Any Bayesian fit folder should contain the following files:

.. code-block:: text

   colibri_fit/
    ├── bayes_metrics.csv           
    ├── filter.yml                  # YAML file: copy of the input runcard
    ├── full_posterior_sample.csv   
    ├── input/                      # directory of input data and runcard(s)
    ├── md5                         # checksum file to verify integrity of the fit folder
    ├── pdf_model.pkl               # pickled PDF model used for the fit
    └── replicas/                   # Folder containing replica sub-folders (one per replica) with exportgrid files.



The ``replicas`` folder contains the subfolders of the replicas that were used in the fit. 
Each of these folders contains an ``.exportgrid`` file, which can be interpreted as a sample 
from the posterior distribution of the PDF model.
The ``pdf_model.pkl`` file contains the pickled PDF model used for the fit. This file can 
be used for several purposes. For example, it can be used to resample from the posterior 
distribution of the PDF model when a Bayesian fit is performed (See also ``colibri.scripts.ns_resampler``).
The ``input/`` directory contains data and the ``filter.yml`` file is a copy of the input 
runcard used for the fit.
The ``md5`` file is a checksum file that can be used to verify the integrity of the fit folder.
The ``bayes_metrics.csv`` file contains the metrics of the fit, such as the log-likelihood
and the evidence.
The ``full_posterior_sample.csv`` file contains the full posterior sample of the fit
(whose size is specified in the runcard). 

Depending on the type of Bayesian fit, other files may be present. For example for a Bayesian fit
using UltraNest, the following files will be present if ``sampler_plot`` is set to ``true``:

.. code-block:: text

   ultranest_colibri_fit/
   ├── ultranest_logs/
   ├── ns_result.csv

While a fit done using the ``analytic_fit`` module will contain the following extra file:

.. code-block:: text

   analytic_colibri_fit/
   ├── analytic_result.csv

Finding the :math:`\chi^2` of a Bayesian Fit
""""""""""""""""""""""""""""""""""""""""""""

The :math:`\chi^2` for a Bayesian fit is stored in the ``bayes_metrics.csv`` file, which looks
like this:

.. code-block:: bash

   bayes_complexity,avg_chi2,min_chi2,logz
   6.693346300122812,3633.618330629202,3.62692e+03,-1.83561e+03

After running a Bayesian fit, you should evolve it as described in :ref:`evolution_script`.

.. _mc_fit_folders:

MC replica fit folders
^^^^^^^^^^^^^^^^^^^^^^

.. note::

    By “MC replica fit folder” we mean a folder containing the results of a fit
    performed with a Monte Carlo replica method (See :cite:`Costantini:2024wby` for more details on this method.).

A MC replica fit folder should have the following structure:

.. code-block:: text

   mc_replica_fit/
    ├── filter.yml           # YAML file: copy of the input runcard
    ├── fit_replicas/        # Folder containing replica sub-folders (one per replica) with exportgrid files.
    ├── input/               # directory of input data and runcard(s)
    ├── md5                  # checksum file to verify integrity of the fit folder
    └── pdf_model.pkl        # pickled PDF model used for the fit

   
where the ``fit_replicas`` folder contains the subfolders of the replicas that were used in the fit.
The other files/folders are analogous to the ones produced by a Bayesian fit, discussed above.

Finding the :math:`\chi^2` of a Monte Carlo fit
"""""""""""""""""""""""""""""""""""""""""""""""

The :math:`\chi^2` for each replica of a MC fit is stored in the
``fit_replicas/replica_n/mc_loss.csv`` file, where `n` is the specific replica number.
This file lists the training and validation losses for every 50 epochs. For example,
the first few lines would look like this:

.. code-block:: bash

   epochs,training_loss,validation_loss
   0,7.80859e+00,1.13569e+01
   1,6.19384e+00,9.22697e+00
   2,4.86740e+00,7.44600e+00
   ...

which would represent the losses for the first 150 epochs (i.e. 0, 1, 2 are just labels).

Postfit selection
"""""""""""""""""

After running a MC fit, you should run a postfit selection of the replicas. This is done by
the ``colibri.scripts.mc_postfit`` script, which uses the `fit_replicas`` and creates a
new ``replicas`` folder, which contains the replicas that pass the postfit, and are the ones
used to evolve the fit.

You can run a postfit selection by running:

.. code-block:: bash

    mc_postfit -c CHI2_THRESHOLD monte_carlo_output_directory 

where the ``-c`` is optional and ``CHI2_THRESHOLD`` is a number that determines
the :math:`\chi^2` threshold above which a MC replica will be rejected, where this
value is taken from the last row of the ``training_loss`` column shown above.
This can also be run as ``--chi2_threshold`` instead of ``-c``. If no value is 
specified, a default value of 1.5 will be applied.

Other options are:

* ``--nsigma NSIGMA``: The nsigma threshold above which replicas are rejected. The default is 5.
* ``--target_replicas TARGET_REPLICAS`` or ``-t TARGET_REPLICAS``: The target number of replicas to be produced by postfit. The default is 100.


After running a postfit selection of a MC fit, you should evolve it as
described in :ref:`this section <evolution_script>`.