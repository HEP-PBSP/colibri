.. _blackjax_fit:

==========================
Bayesian Fit with BlackJAX
==========================

In this tutorial, we will look at an example runcard to run a Bayesian fit with the
BlackJAX nested sampler :cite:`cabezas2024blackjax, yallup2025nested`. We will do
so for the Les Houches parametrisation model (see :ref:`this tutorial <in_les_houches>`
for details on the Les Houches model and how to implement it).

An advantage of using BlackJAX is that it can run entirely on GPU (as well as on CPU),
so we will discuss the optimal settings to do so.

We will then look at the command to execute the runcard.

.. _blackjax_runcard:

Runcard
-------

The runcard ``runcards/lh_blackjax.yaml`` in the ``examples/les_houches_example``
directory runs a BlackJAX fit of the Les Houches model.

.. literalinclude:: ../../../../../../examples/les_houches_example/runcards/lh_blackjax.yaml
    :language: yaml

Note that the ``prior_settings`` are the same as in an :ref:`UltraNest fit <ultranest_fit>`,
and so all the settings described there can be used (e.g. global bounds).

``blackjax_settings``
^^^^^^^^^^^^^^^^^^^^^

* ``n_posterior_samples``: Number of posterior samples ('replicas') drawn (*resampled*) from the posterior distribution. The default is 1000. See :ref:`this tutorial <resampling_script>` for details on resampling.
* ``n_live``: Number of live points at any given time. More live points results in a better estimate of the error.
* ``repeats``: Number of slice-sampling steps per update, in units of the number of parameters (the sampler takes ``repeats`` x ``n_params`` inner steps).
* ``delete_fraction``: Fraction of live points allowed to be deleted. The more deleted points, the higher the risk of getting stuck at a local minimum, but the lower the memory usage. This setting is analogous to ``min_live_points`` in an :ref:`UltraNest fit <ultranest_fit>`, in that a ``delete_fraction`` of 0.5 is equivalent to 250 ``min_live_points``.
* ``log_precision``: Termination ratio. This setting is analogous to ``frac_remain`` in an :ref:`UltraNest fit <ultranest_fit>`, in that a ``log_precision`` of -3 would be equivalent to a ``frac_remain`` of 0.001.
* ``sampler_plot``: ``true`` produces diagnostic plots of the posterior.
* ``posterior_resampling_seed``: Random seed used when resampling posterior samples. Fixing this seed ensures reproducible posterior replicas for a given nested sampling run.
* ``blackjax_seed``: Global random seed for the BlackJAX nested sampler. Setting this seed makes the nested sampling reproducible.

Running on GPU
^^^^^^^^^^^^^^

BlackJAX nested sampling is designed to run entirely in memory on a GPU, which
avoids slow CPU↔GPU transfers, allowing accelerated fits. In order to run on a GPU,
you may want to adjust the settings to take into account the following points:

*  ``n_live`` can be made much larger than in a CPU run, namely of the order of ~2000-10000, affording more accurate calculations of model evidences.
* The number ``n_live`` x ``delete_fraction`` is effectively the number of Markov Chains running in parallel. For example, a ``delete_fraction`` of 0.5 and a ``n_live`` 0f 5000 does 2500 updates in parallel. A larger number of parallel updates is advantageous provided the throughput of the GPU isn't saturated.
* The algorithm natively supports single precision numerics, which can help if engineering for fast GPU throughput.

You can check the point at which the GPU becomes saturated by timing the
computation of the likelihood, which is shown in :ref:`this tutorial <time_likelihood>`.

Running the fit
---------------

In general, Colibri runcards can be executed by running the following command:

.. code-block:: bash

    model_executable runcard.yaml

This must be done after installing the dependencies specific to the model. For 
example, for the Les Houches parametrisation model presented in 
:ref:`this tutorial <in_les_houches>`, the first step would be to run 

.. code-block:: bash

    pip install -e .

from the ``examples/les_houches_example`` directory.

Then, from the same directory, you can run the above runcard with the following command:

.. code-block:: bash

    les_houches_exe runcards/lh_blackjax.yaml


Running fits will generate fit folders, the details of which can be found in 
:ref:`this section <bayes_fit_folders>`.

The next step would be to :ref:`evolve your fit <evolution_script>`.