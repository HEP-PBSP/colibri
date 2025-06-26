
.. _ultranest_fit:

===========================
Bayesian Fit with UltraNest
===========================

We will first look at an example runcard to run a Bayesian fit with the Les Houches
parametrisation model (see :ref:`this tutorial <in_les_houches>` for details
on the Les Houches model and how to build it).

Then we will look at the command to execute the runcard. 

Runcard
-------

.. code-block:: bash

    meta: 'An example fit using Colibri, reduced DIS dataset.'

    #######################
    # Data and theory specs
    #######################

    dataset_inputs:    
    # DIS          
    - {dataset: SLAC_NC_NOTFIXED_P_EM-F2, variant: legacy_dw}
    - {dataset: SLAC_NC_NOTFIXED_D_EM-F2, variant: legacy_dw}
    - {dataset: BCDMS_NC_NOTFIXED_P_EM-F2, variant: legacy_dw}
    - {dataset: BCDMS_NC_NOTFIXED_D_EM-F2, variant: legacy_dw}
    # - {dataset: CHORUS_CC_NOTFIXED_PB_NU-SIGMARED, variant: legacy_dw}
    # - {dataset: CHORUS_CC_NOTFIXED_PB_NB-SIGMARED, variant: legacy_dw}
    # - {dataset: NUTEV_CC_NOTFIXED_FE_NU-SIGMARED, cfac: [MAS], variant: legacy_dw}
    # - {dataset: NUTEV_CC_NOTFIXED_FE_NB-SIGMARED, cfac: [MAS], variant: legacy_dw}
    # - {dataset: HERA_NC_318GEV_EM-SIGMARED, variant: legacy}
    # - {dataset: HERA_NC_225GEV_EP-SIGMARED, variant: legacy}
    # - {dataset: HERA_NC_251GEV_EP-SIGMARED, variant: legacy}
    # - {dataset: HERA_NC_300GEV_EP-SIGMARED, variant: legacy}
    # - {dataset: HERA_NC_318GEV_EP-SIGMARED, variant: legacy}
    # - {dataset: HERA_CC_318GEV_EM-SIGMARED, variant: legacy}
    # - {dataset: HERA_CC_318GEV_EP-SIGMARED, variant: legacy}
    # - {dataset: HERA_NC_318GEV_EAVG_CHARM-SIGMARED, variant: legacy}
    # - {dataset: HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED, variant: legacy}
    # - {dataset: NMC_NC_NOTFIXED_EM-F2, variant: legacy_dw}
    # - {dataset: NMC_NC_NOTFIXED_P_EM-SIGMARED, variant: legacy}

    theoryid: 40000000                     # The theory from which the predictions are drawn.
    use_cuts: internal                     # The kinematic cuts to be applied to the data.

    #####################
    # Loss function specs
    #####################

    positivity:                            # Positivity datasets, used in the positivity penalty.
    posdatasets:
    - {dataset: NNPDF_POS_2P24GEV_F2U, variant: None, maxlambda: 1e6}

    positivity_penalty_settings:
        positivity_penalty: false
        alpha: 1e-7                           
        lambda_positivity: 0                 

    # Integrability Settings
    integrability_settings:
        integrability: False            

    use_fit_t0: True                        # Whether the t0 covariance is used in the chi2 loss.
    t0pdfset: NNPDF40_nnlo_as_01180         # The t0 PDF used to build the t0 covariance matrix.
    

    ###################
    # Methodology specs
    ###################
    prior_settings:
    prior_distribution: uniform_parameter_prior
    prior_distribution_specs:
        bounds:
            alpha_gluon: [-0.1, 1]
            beta_gluon: [9, 13]
            alpha_up: [0.4, 0.9]
            beta_up: [3, 4.5]
            epsilon_up: [-3, 3]
            gamma_up: [1, 6]
            alpha_down: [1, 2]
            beta_down: [8, 12]
            epsilon_down: [-4.5, -3]
            gamma_down: [3.8, 5.8]
            norm_sigma: [0.1, 0.5]
            alpha_sigma: [-0.2, 0.1]
            beta_sigma: [1.2, 3]


    # Nested Sampling settings
    ns_settings:
        sampler_plot: true
        n_posterior_samples: 100        # Number of posterior samples generated.
        ReactiveNS_settings:
            vectorized: False
            ndraw_max: 500              # Maximum number of points to simultaneously propose.
        Run_settings:
            min_num_live_points: 200    # Minimum number of live points throughout the run.
            min_ess: 50                 # Target number of effective posterior samples.
            frac_remain: 0.3            # Integrate until this fraction of the integral is left in the remainder. 
        SliceSampler_settings:
            nsteps: 106                 # number of accepted steps until the sample is considered independent.


    actions_:
    - run_ultranest_fit                 # Choose from ultranest_fit, monte_carlo_fit, analytic_fit

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

``ns_settings``
^^^^^^^^^^^^^^^

* ``sampler_plot``: ``true`` will generate diagnostic plots (corner, run and trace plots)
    in ``fit_output_directory/ultranest_logs/plots``. These help assess the convergence
    and efficiency of the fit.
* ``n_posterior_samples``: Number of posterior samples drawn from the posterior distribution.
* ``vectorized``: Determines whether the likelihood function supports vectorised evaluation
    (i.e., evaluating multiple points at once).
* ``ndraw_max``: Maximum number of points to simultaneously propose. Can be commented out.
* ``min_num_live_points``: Minimum number of live points throughout the run.
* ``min_ess``: Target number of effective posterior samples.
* ``frac_remain``: Integrate until this fraction of the integral is left in the remainder. 
* ``SliceSampler_settings``: Sampling uniformly within "slices" of constant probability.
    Slice sampling is optional, so these settings can be commented out.
* ``nsteps``: Number of accepted steps until the sample is considered independent.


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

Then, you can use the above runcard with the following command:

.. code-block:: bash

    les_houches_exe runcard.yaml


Running fits will generate fit folders, the details of which can be found in 
:ref:`this section <bayes_fit_folders>`.

Terminal output
^^^^^^^^^^^^^^^

As the fit runs, a **status line** and **live point display** will be displayed
in the terminal for each iteration. For details on what they mean and how to 
interpret them, see the 
`UltraNest documentation <https://johannesbuchner.github.io/UltraNest/index.html#>`_.
Specifically, `this page <https://johannesbuchner.github.io/UltraNest/issues.html>`_.