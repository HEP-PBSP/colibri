.. _les_houches:

======================
Evolving a Colibri Fit
======================

In this tutorial we will discuss what the general structure of a colibri fit folder is and how it can be evolved.

Colibri fit folders
--------------------
A colibri fit folder is the folder resulting from a colibri-model fit. It is essentially a folder containing a set 
of relevant information for the fit.
Currently, we distinguish between two types of fit folders: Bayesian fit folders and Monte Carlo replica fit folders.


Bayesian fit folders
^^^^^^^^^^^^^^^^^^^^

.. note::

   By “Bayesian fit folder” we mean a folder containing the results of a fit
   performed with a Bayesian sampling method.

Any Bayesian fit folder should contain the following files:

.. code-block:: text

   colibri_fit/
   ├── replicas/         # Folder containing replica sub‐folders (one per replica) with exportgrid files.
   ├── pdf_model.pkl     # pickled PDF model used for the fit
   ├── input/            # directory of input data and runcard(s)
   ├── filter.yml        # YAML file: copy of the input runcard
   └── md5               # checksum file to verify integrity of the fit folder
   ├── bayes_metrics.csv  
   ├── full_posterior_sample.csv


The ``replicas`` folder contains the subfolders of the replicas that were used in the fit. 
Each of these folders contains an ``.exportgrid`` file, which can be interpreted as a sample from the posterior distribution 
of the PDF model.
The ``pdf_model.pkl`` file contains the pickled PDF model used for the fit. This file can be used for several purposes,
an example is that of using it to resample from the posterior distribution of the PDF model when a Bayesian fit is performed
(See also `colibri.scripts.ns_resampler`).
The other files are the input data and the filter file, which is a copy of the input runcard used for the fit.
The ``md5`` file is a checksum file that can be used to verify the integrity of the fit folder.
The ``bayes_metrics.csv`` file contains the metrics of the fit, such as the log-likelihood and the evidence.
The ``full_posterior_sample.csv`` file contains the full posterior sample of the fit (whose size is specified in the runcard). 

Depending on the type of Bayesian fit, other files may be present, for example a fit done using the 
ultranest will contain the following extra files:

.. code-block:: text

   ultranest_colibri_fit/
   ├── ultranest_logs/
   ├── ns_result.csv

While a fit done using the `analytic_fit` module will contain the extra following file:

.. code-block:: text

   analytic_colibri_fit/
   ├── analytic_result.csv


MC replica fit folders
^^^^^^^^^^^^^^^^^^^^^^

.. note::

    By “MC replica fit folder” we mean a folder containing the results of a fit
    performed with a Monte Carlo replica method (See :cite:`Costantini:2024wby` for more details on this method.).

A MC replica fit folder should have the following structure:

.. code-block:: text

   mc_replica_fit/
   ├── fit_replicas/         # Folder containing replica sub‐folders (one per replica) with exportgrid files.
   ├── pdf_model.pkl     # pickled PDF model used for the fit
   ├── input/            # directory of input data and runcard(s)
   ├── filter.yml        # YAML file: copy of the input runcard
   └── md5               # checksum file to verify integrity of the fit folder
   
where the ``fit_replicas`` folder contains the subfolders of the replicas that were used in the fit.
This subfolder, in particular, is used by the `colibri.scripts.mc_postfit` script to 
perform a postfit selection of the replicas. The postfit script also takes care of creating 
the ``replicas`` folder, which is the one needed for the evolution of the fit.


Evolution script
-----------------

The evolution script of colibri is a wrapper around the `evolven3fit` script
(See the :mod:`colibri.scripts.evolve_fit` module’s and :func:`colibri.scripts.evolve_fit.main` function.)
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

.. note::

    The postfit emulation is only run for bayesian fits and the script will look into the fit folder to check
    whether a `bayes_metrics.csv` file is present. If it is not, the script will not run the postfit emulation.

After running the evolution script, it is possible (if the user has the right permissions) to simply upload the fit
to the `validphys` server using the validphys script

.. code-block:: bash

   vp-upload <name_fit>

After which the fit can be installed and made available in the environment with the command

.. code-block:: bash

   vp-get fit <name_fit>

If the user does not have the right permissions it is recommended to simply symlink the lhapdf set to the 
lhapdf environment folder.


in order to be able to use ``validphys`` reports for 
fits we   
