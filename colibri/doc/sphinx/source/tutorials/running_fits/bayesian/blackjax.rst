.. _blackjax_fit:

==========================
Bayesian Fit with BlackJAX
==========================

In this tutorial, we will look at an example runcard to run a Bayesian fit with the
BlackJAX nested sampler :cite:`cabezas2024blackjax, yallup2025nested`. We will do
so for the Les Houches parametrisation model (see :ref:`this tutorial <in_les_houches>`
for details on the Les Houches model and how to implement it).

We will then loot at the command to execute the runcard.

.. _blackjax_runcard:

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

    closure_test_level: 0                  # The closure test level: False for experimental, level 0
                                        # for pseudodata with no noise, level 1 for pseudodata with
                                        # noise.

    closure_test_pdf: LH_PARAM_20250519  # The closure test PDF used if closure_test_level is not False

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

    use_fit_t0: True                       # Whether the t0 covariance is used in the chi2 loss.
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
    blackjax_settings:
        n_posterior_samples: 100
        vectorized: True
        n_live: 500
        repeats: 3
        delete_fraction: 0.5
        log_precision: -3
        seed: 0



    actions_:
    - run_blackjax_fit                      # Choose from ultranest_fit, monte_carlo_fit, analytic_fit

Note that the ``prior_settings`` are the same as in an :ref:`UltraNest fit <ultranest_fit>`,
and so all the settings described there can be used (e.g. global bounds).

``blackjax_settings``
^^^^^^^^^^^^^^^^^^^^^

* ``n_posterior_samples``: Number of posterior samples ('replicas') drawn (*resampled*) from the posterior distribution. The default is 1000. See :ref:`this tutorial <resampling_script>` for details on resampling.
* ``vectorized``: Determines whether the likelihood function supports vectorised evaluation (i.e., evaluating multiple points at once).
* ``n_live``: Number of live points at any given time. More live points results in a better estimate of the error.
* ``repeats``: Number of successful Monte Carlo steps required. Should be a multiple of the dimentionality of parameter space. 
* ``delete_fraction``: Fraction of live points allowed to be deleted. The more deleted points, the higher the risk of getting stuck at a local minimum, but the lower the memory usage. This setting is analogous to ``min_live_points`` in an :ref:`UltraNest fit <ultranest_fit>`, in that a ``delete_fraction`` of 0.5 is equivalent to 250 ``min_live_points``.
* ``log_precision``: Termination ratio. This setting is analogous to ``frac_remain`` in an :ref:`UltraNest fit <ultranest_fit>`, in that a ``log_precision`` of -3 would be equivalent to a ``frac_remain`` of 0.01.
* ``seed``: 

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

The next step would be to :ref:`evolve your fit <evolution_script>`.