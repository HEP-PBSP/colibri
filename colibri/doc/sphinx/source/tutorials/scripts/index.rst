.. _evolution:

===============
Colibri Scripts
===============

In this tutorial we will discuss what the general structure of a colibri fit folder is 
and some of the scripts that are available in colibri.

.. _colibri_fit_folders:

Colibri fit folders
-------------------
A colibri fit folder is the folder resulting from a colibri-model fit. It is essentially 
a folder containing a set of relevant information for the fit.
Currently, we distinguish between two types of fit folders: Bayesian fit folders and 
Monte Carlo replica fit folders.

.. _bayes_fit_folders:

Bayesian fit folders
^^^^^^^^^^^^^^^^^^^^

.. note::

   By “Bayesian fit folder” we mean a folder containing the results of a fit
   performed with a Bayesian sampling method (see :ref:`this section <running_bayesian>` for details on how to run a
   Bayesian fit).

Any Bayesian fit folder should contain the following files:

.. code-block:: text

   colibri_fit/
   ├── replicas/         # Folder containing replica sub‐folders (one per replica) with exportgrid files.
   ├── pdf_model.pkl     # pickled PDF model used for the fit
   ├── input/            # directory of input data and runcard(s)
   ├── filter.yml        # YAML file: copy of the input runcard
   ├── md5               # checksum file to verify integrity of the fit folder
   ├── bayes_metrics.csv  
   └── full_posterior_sample.csv


The ``replicas`` folder contains the subfolders of the replicas that were used in the fit. 
Each of these folders contains an ``.exportgrid`` file, which can be interpreted as a sample 
from the posterior distribution of the PDF model.
The ``pdf_model.pkl`` file contains the pickled PDF model used for the fit. This file can 
be used for several purposes,an example is that of using it to resample from the posterior 
distribution of the PDF model when a Bayesian fit is performed (See also `colibri.scripts.ns_resampler`).
The other files are the input data and the filter file, which is a copy of the input 
runcard used for the fit.
The ``md5`` file is a checksum file that can be used to verify the integrity of the fit folder.
The ``bayes_metrics.csv`` file contains the metrics of the fit, such as the log-likelihood
and the evidence.
The ``full_posterior_sample.csv`` file contains the full posterior sample of the fit
(whose size is specified in the runcard). 

Depending on the type of Bayesian fit, other files may be present, for example a fit done 
using the ultranest will contain the following extra files:

.. code-block:: text

   ultranest_colibri_fit/
   ├── ultranest_logs/
   ├── ns_result.csv

While a fit done using the `analytic_fit` module will contain the extra following file:

.. code-block:: text

   analytic_colibri_fit/
   ├── analytic_result.csv

.. _mc_fit_folders:

MC replica fit folders
^^^^^^^^^^^^^^^^^^^^^^

.. note::

    By “MC replica fit folder” we mean a folder containing the results of a fit
    performed with a Monte Carlo replica method (See :cite:`Costantini:2024wby` for more details on this method.).

A MC replica fit folder should have the following structure:

.. code-block:: text

   mc_replica_fit/
   ├── fit_replicas/     # Folder containing replica sub‐folders (one per replica) with exportgrid files.
   ├── pdf_model.pkl     # pickled PDF model used for the fit
   ├── input/            # directory of input data and runcard(s)
   ├── filter.yml        # YAML file: copy of the input runcard
   └── md5               # checksum file to verify integrity of the fit folder
   
where the ``fit_replicas`` folder contains the subfolders of the replicas that were used in the fit.
This subfolder, in particular, is used by the `colibri.scripts.mc_postfit` script to 
perform a postfit selection of the replicas. The postfit script also takes care of creating 
the ``replicas`` folder, which is the one needed for the evolution of the fit.

To run the postfit script, you should run the following command:

You can then run a postfit selection of the replicas by running

.. code-block:: bash

    mc_postfit -c CHI2_THRESHOLD monte_carlo_output_directory 

where the ``-c `` is optional and ``CHI2_THRESHOLD`` is a number that determines
the :math:`\chi^2` threshold above which a Monte Carlo replica will be rejected.
This can also be run as ``--chi2_threshold`` instead of ``-c``.

Other options are:

* ``--nsigma NSIGMA``: The nsigma threshold above which replicas are rejected.
* ``--target_replicas TARGET_REPLICAS`` or ``-t TARGET_REPLICAS``: The target number of replicas to be produced by postfit.

Evolution script
----------------

The evolution script of colibri is a wrapper around the `evolven3fit` script
(See the :mod:`colibri.scripts.evolve_fit` module's and :func:`colibri.scripts.evolve_fit.main` function.)
that only allows for the `evolve` option. 

It can be executed from the command line as follows:

.. code-block:: bash

   evolve_fit <name_fit>

where ``<name_fit>`` is the name of the fit you want to evolve.
The script also has a ``--help`` option that will show you all the options available.
For more information on the evolution see also the helper from the ``evolven3fit`` script.

Postfit emulation
^^^^^^^^^^^^^^^^^

For Bayesian fits we don't do any postfit selection on the posterior, however, for backwards compatibility with the 
`validphys` module we still run a postfit emulation which takes care of creating the central replica and a `postfit` 
folder containing the evolved replicas as well as the corresponding LHAPDF set.

Upload of the fit
^^^^^^^^^^^^^^^^^

After running the evolution script, it is possible (if the user has the right permissions) to simply upload the fit
to the `validphys` server using the validphys script

.. code-block:: bash

   vp-upload <name_fit>

After which the fit can be installed and made available in the environment with the command

.. code-block:: bash

   vp-get fit <name_fit>

If the user does not have the right permissions it is recommended to simply symlink the lhapdf set to the 
lhapdf environment folder or to symlink the fit folder to the `NNPDF/results` folder of the environment.

.. note::

    The final folder after the evolution will also contain a symlink `nnfit -> replicas` needed for `validphys` and 
    `evolven3fit` as well as a `postfit` folder.


Resampling script
-----------------

In a Colibri fit runcard, you control how many posterior samples get written out as .exportgrid files in the 
``replicas/`` folder — and those can subsequently be evolved into a PDF set.

For a Bayesian fit using the analytical - inference method, set the total number of posterior draws via the 
``analytic_settings`` block. For example:

.. code-block:: yaml

    # Analytic settings
    analytic_settings:
      n_posterior_samples: 100
      full_sample_size: 50000

Likewise, if you instead use the UltraNest nested sampler, specify exactly the same parameter name under 
``ultranest_settings``:

.. code-block:: yaml

    # ultranest settings
    ultranest_settings:
      n_posterior_samples: 100
      ...


**Key Parameters**


- ``n_posterior_samples``: 
  The number of individual posterior draws that will each be written out as a separate
  ``.exportgrid`` file in the ``replicas/`` folder.

- ``full_sample_size`` *(analytic only)* : 
  The total size of the merged posterior sample, which is saved to
  ``full_posterior_sample.csv`` at the top level of your fit directory.

.. note::
    
    In the case of a fit done using the ``ultranest`` nested sampling sampler, 
    the ``full_sample_size`` defaults to an internal number that might depends on the 
    specific run.


If you want to draw additional replicas (or have a smaller set for a finite-size effects studies) from the posterior distribution 
of an already‐completed PDF fit, you do **not** need to re‐run the full fit. 
Instead, use the ``resample_fit`` helper script.

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
    
    Importantly, in order to resample from the posterior distribution of a fit, you need
    to be in the same environment as the one used to perform the fit.
    Hence, if you want to resample a fit done using the ``les-houches`` PDF model, you need to 
    be in the environment where the ``les_houches_exe`` exectuable is available.