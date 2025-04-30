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

Both folders contain the following files:

.. code-block:: text

   colibri_fit/
   ├── replicas/         # Folder containing replica sub‐folders (one per replica) with exportgrid files.
   ├── pdf_model.pkl     # pickled PDF model used for the fit
   ├── input/            # directory of input data and runcard(s)
   ├── filter.yml        # YAML file: copy of the input runcard
   └── md5               # checksum file to verify integrity of the fit folder

The ``replicas`` folder contains the subfolders of the replicas that were used in the fit. 
Each of these folders contains an ``.exportgrid`` file, which can be interpreted as a sample from the posterior distribution 
of the PDF model.
The ``pdf_model.pkl`` file contains the pickled PDF model used for the fit. This file can be used for several purposes,
an example is that of using it to resample from the posterior distribution of the PDF model when a Bayesian fit is performed
(See also `colibri.scripts.ns_resampler`).
The other files are the input data and the filter file, which is a copy of the input runcard used for the fit.
The ``md5`` file is a checksum file that can be used to verify the integrity of the fit folder.

Bayesian fit folders
^^^^^^^^^^^^^^^^^^^^

.. note::

   By “Bayesian fit folder” we mean a folder containing the results of a fit
   performed with a Bayesian sampling method.

On top of the files mentioned above, a Bayesian fit folder should also contains the following files:

.. code-block:: text

   bayesian_colibri_fit/
   ├── bayes_metrics.csv       
   ├── full_posterior_sample.csv

Moreover, depending on the type of Bayesian fit, other files may be present, for example a fit done using the 
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
For more information on the evolution see also the helper of the ``evolven3fit`` script.

Postfit emulation
-----------------
Bayesian fits don't need a Postfit operation, however, in order to be able to use ``validphys`` reports for 
fits we   

Fit folder structure
--------------------
After the  the fit a 
   


