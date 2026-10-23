.. _ultranest_fit:

===========================
Bayesian Fit with UltraNest
===========================

In this tutorial we will first look at an example runcard to run a Bayesian fit
with the Les Houches parametrisation model (see :ref:`this tutorial <in_les_houches>`
for details on the Les Houches model and how to implement it).

Then we will look at the command to execute the runcard. 

We do this using UltraNest as the nested sampler
:cite:`Buchner2016,Buchner2019,Buchner2021`.

.. _ultranest_runcard:

Runcard
-------

The runcard ``runcards/lh_fit_closure_test.yaml`` in the ``examples/les_houches_example``
directory runs an UltraNest fit of the Les Houches model.

.. literalinclude:: ../../../../../../examples/les_houches_example/runcards/lh_fit_closure_test.yaml
    :language: yaml

Note how the prior bounds need to be specified for each parameter. Alternatively, 
global bounds (i.e the same bounds for all parameters) can be used, by replacing 

.. code-block:: bash

    bounds:
        alpha_gluon: [-0.1, 1]
        beta_gluon: [9, 13]
    ...

with, for example: 

.. code-block:: bash
    
    min_val: -4.5
    max_val: 13

in those cases where it is appropriate for the given parameters of the model 
(eg. only one parameter or all parameters have close numerical values).

For details on general settings (such as ``positivity``) see
:ref:`this section <general_settings>`.

``ultranest_settings``
^^^^^^^^^^^^^^^^^^^^^^
* ``ultranest_seed``:  Seed for the numpy random number generator used by UltraNest.
* ``sampler_plot``: ``true`` will generate diagnostic plots (corner, run and trace plots) in ``fit_output_directory/ultranest_logs/plots``. These help assess the convergence and efficiency of the fit.
* ``n_posterior_samples``: Number of posterior samples ('replicas') drawn (*resampled*) from the posterior distribution. The default is 1000. See :ref:`this tutorial <resampling_script>` for details on resampling.
* ``vectorized``: Determines whether the likelihood function supports vectorised evaluation (i.e., evaluating multiple points at once).
* ``ndraw_max``: Maximum number of points to simultaneously propose. Can be commented out.
* ``min_num_live_points``: Minimum number of live points throughout the run.
* ``min_ess``: Target number of effective posterior samples.
* ``frac_remain``: Integrate until this fraction of the integral is left in the remainder. 
* ``SliceSampler_settings``: Sampling uniformly within "slices" of constant probability. Slice sampling is optional, so these settings can be commented out. 
* ``nsteps``: Number of accepted steps until the sample is considered independent.
* ``posterior_resampling_seed``: Seed for resampling the posterior samples.

``prior_distribution``
^^^^^^^^^^^^^^^^^^^^^^
As well as initialising with a uniform prior distribution, such as in the runcard
above, it is possible to initialise by setting the prior from a gaussian posterior
as a result from a previous fit. This has shown to yield equivalent results, while
being more computationally efficient (see Ref. :cite:alp:`Costantini:2025wxp`).
This can be achieved by setting:

.. code-block:: bash

    prior_settings:
        prior_distribution: prior_from_gauss_posterior
        prior_distribution_specs:
            prior_fit: your_previous_fit

The previous fit ``your_previous_fit`` folder needs to be placed in ``sys.prefix/share/colibri/results/``.

UltraNest settings
^^^^^^^^^^^^^^^^^^
For ``ReactiveNS_settings`` and ``Run_settings``, you can use any of the options of
UltraNest's ``ReactiveNestedSampler``, which you can read more about
`here <https://johannesbuchner.github.io/UltraNest/ultranest.html#>`_.

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

    les_houches_exe runcards/lh_fit_closure_test.yaml


Running fits will generate fit folders, the details of which can be found in 
:ref:`this section <bayes_fit_folders>`.

The next step would be to :ref:`evolve your fit <evolution_script>`.

Terminal output
^^^^^^^^^^^^^^^

As the fit runs, a **status line** and **live point display** will be displayed
in the terminal for each iteration. For details on what they mean and how to 
interpret them, see the 
`UltraNest documentation <https://johannesbuchner.github.io/UltraNest/index.html#>`_.
Specifically, `this page <https://johannesbuchner.github.io/UltraNest/issues.html>`_.