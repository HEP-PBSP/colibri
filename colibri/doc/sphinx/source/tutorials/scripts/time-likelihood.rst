.. _time_likelihood:

=========================
Timing the log-Likelihood
=========================

In this tutorial, we show how to time the log-likelihood for different
batch sizes of the vectorised likelihood. This can help you understand
what the computational cost is for the computation of the log-likelihood
for your machine and for a given model.

The runcard ``runcards/lh_time_likelihood.yaml`` in the ``examples/les_houches_example``
directory times the log-likelihood for the :ref:`Les Houches parametrisation model <lh_theory>`.

.. literalinclude:: ../../../../../examples/les_houches_example/runcards/lh_time_likelihood.yaml
    :language: yaml

The ``batch_sample_sizes`` is optional. If it is not specified, default values of
``[1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]`` will be used. These are
the batch sizes for which the likelihood will be timed.

You can run the script with:

.. code-block:: bash

    les_houches_exe runcards/lh_time_likelihood.yaml

from the ``examples/les_houches_example`` directory. Note that we are using 
a model-specific executable (``les_houches_exe``). You should therefore adapt
``param_initialiser_settings`` in the runcard above to your specific model. 
In general, the time it takes to compute the likelihood is dependent on the model
that the likelihood is computed for. For example, it would take loner for a model
with more parameters, ect. 

This script will produce a file called ``log_likelihood_times.csv``, that looks as
follows:

.. code-block:: bash

    batch_size,avg_time_seconds,relative_time
    1,0.00024411916732788085,1.0
    10,0.00024394989013671876,0.9993065796798547
    100,0.0035084390640258787,14.371829555332011

    [...]

where the ``relative_time`` is the time with respect to the smallest batch size.
