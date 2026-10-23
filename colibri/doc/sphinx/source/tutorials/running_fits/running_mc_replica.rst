.. _running_mc_replica:


===============================
Monte Carlo Replica Method Fits
===============================

In this section, we describe how to run a Monte Carlo (MC) replica method fit
in Colibri.

The Monte Carlo replica method introduces a `pseudodata distribution`,
:math:`d_p \sim N(d_0, \Sigma)`, which is an approximation to the actual distribution
from which the measurement :math:`d_0` was drawn. It then defines the corresponding
'best-fit parameter' values to be those which minimise the :math:`\chi^2` evaluated
on the pseudodata:

.. math::

    \chi^2 = (d_p - t(\theta))^T \, \Sigma^{-1} \, (d_p - t(\theta))

where:

- :math:`d_p` is the pseudodata vector, drawn from the distribution :math:`d_p \sim N(d_0, \Sigma)`,
- :math:`t(\theta)` is the theoretical prediction vector, dependent on the model parameters :math:`\theta`,
- :math:`\Sigma` is the covariance matrix encoding the uncertainties.

The idea is to repeat this procedure across multiple data replicas, so that the collection of best-fit 
models for each pseudodata instance approximates a posterior sample.

Runcard
-------

The runcard ``runcards/lh_monte_carlo.yaml`` in the ``examples/les_houches_example``
directory performs a Monte Carlo replica fit of the Les Houches model. Note that the
dependence on the model comes from the model-specific executable.

.. literalinclude:: ../../../../../examples/les_houches_example/runcards/lh_monte_carlo.yaml
    :language: yaml

``optimizer_settings``
^^^^^^^^^^^^^^^^^^^^^^
These settings control the method of gradient descent. You can use any
of the Optax optimizers and settings, which you can read more about
`here <https://optax.readthedocs.io/en/latest/api/optimizers.html#>`__.
Learning schedulers are also supported, and you can find the available options
`here <https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#>`__.

.. _param-initialiser-settings:

``param_initialiser_settings``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If it is appropriate for a given model, you may choose to have single, 
global minimum and maximum values for all parameters, instead of specific
bounds for each parameter. In that, case, you may replace

.. code-block:: bash

    bounds:
        alpha_gluon: [-0.1, 1]
        beta_gluon: [9, 13]
    ...

with, for example: 

.. code-block:: bash
    
    min_val: -4.5
    max_val: 13

Initialising parameters with a normal distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to initialise with parameters that are distributed
normally rather than uniformally (as in the runcard above) you
may do so with the following settings:

.. code-block:: bash

    param_initialiser_settings:               # The initialiser for Monte Carlo training.
        type: normal
        means: 2.0    
        stds: 0.5

This will set the mean to 2.0 and standard deviation to 0.5 for all
parameters. If instead you want to specify different values for each
parameter, either for the means or standard deviations or both, you
can do so:

.. code-block:: bash

    param_initialiser_settings:               # The initialiser for Monte Carlo training.
        type: normal
        means:
            alpha_gluon: 0.356
            beta_gluon: 10.9      
            [...]
        stds:
            alpha_gluon: 0.06    
            beta_gluon: 0.55              
            [...]

.. note::
    If you use initialise with a normal distribution and don't specify
    means or standard deviations, default values of **0.0** and **1.0**
    will be used respectively.


Using data batching
^^^^^^^^^^^^^^^^^^^
In Monte Carlo replica fits, it is possible to use data batching during training.
This can be done by adding the following settings to the runcard:

.. code-block:: bash

    batch_size: 128
    batch_seed: 3
    shuffle_each_epoch: False
    
* ``batch_size``: The size of the data batches. If the dataset is smaller
  than this size or the ``batch_size`` is not specified, the full dataset will be used.
* ``batch_seed``: The random seed used to generate the data batches.
* ``shuffle_each_epoch``: Whether to reshuffle the data at the start of each epoch. If
  set to ``False``, the data will be shuffled only once at the start of training and the batches will be fixed.

Running the fit
---------------

To perform a Monte Carlo fit for the Les Houches model 
(presented in :ref:`this tutorial <in_les_houches>`), you would run:

.. code-block:: bash

    pip install -e .

from the ``examples/les_houches_example`` directory.

Then, from the same directory, you can run the above runcard with the following command:

.. code-block:: bash

    les_houches_exe runcards/lh_monte_carlo.yaml -rep N

Note that this command will generate one single replica, namely replica number
``N``. For a fit with more than one replica, you should iterate the above or 
submit the job to a batch system. 

You can then run a postfit selection on the fit folders and evolve the fit.
Details on how to do this can be found in :ref:`this section <mc_fit_folders>`.

