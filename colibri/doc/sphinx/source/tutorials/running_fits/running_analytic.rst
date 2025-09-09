.. _running_analytic:

===============
Analytical Fits
===============

This section describes how to run an analytic fit with Colibri.

In an analytic fit, the posterior mean and covariance of the PDF model
parameters are obtained via the analytic solution to a linear regression problem.
This fitting methodology is therefore only appropriate for linear models, which,
in general, can be described by an equation of the following form:

.. math::

    f(\theta) = W\,\theta,

where :math:`W` is a matrix that maps the parameters, :math:`\theta`, to the
theory prediction vector, :math:`f(\theta)`. 

Effectively, the analytic fit is treated as a "bayesian fit", where the priors of
the parameters are used to compute the evidence analytically. See
:ref:`this section <th_analytic_fits>` for an overview of the theoretical
background of the analytic fit methodology, as well as details on when it might
be appropriate to run an analytic fit.

In this tutorial, we will provide an example that was implemented in the work
presented in Ref. :cite:alp:`Costantini:2025wxp`.


Runcard
-------

The following runcard can be used to run an analytic fit. This has been used to
fit the weight minimisation (``wmin``) model presented in Ref.
:cite:alp:`Costantini:2025wxp`. 

.. code-block:: bash

    meta: 'An example Level 1 closure test analytic fit using Colibri'

    #######################
    # Data and theory specs
    #######################

    dataset_inputs:
        # DIS data
        # NMC experiment
        # - {dataset: NMC_NC_NOTFIXED_P_EM-SIGMARED, variant: legacy} # (out-of-sample)
        # NUCLEAR experiments
        - {dataset: CHORUS_CC_NOTFIXED_PB_NU-SIGMARED, variant: legacy_dw}
        - {dataset: CHORUS_CC_NOTFIXED_PB_NB-SIGMARED, variant: legacy_dw}
        - {dataset: NUTEV_CC_NOTFIXED_FE_NU-SIGMARED, cfac: [MAS], variant: legacy_dw}
        # - {dataset: NUTEV_CC_NOTFIXED_FE_NB-SIGMARED, cfac: [MAS], variant: legacy_dw} # (out-of-sample)

        # HERACOMB experiments
        - {dataset: HERA_NC_318GEV_EM-SIGMARED, variant: legacy}
        # - {dataset: HERA_NC_225GEV_EP-SIGMARED, variant: legacy} # (out-of-sample)
        - {dataset: HERA_NC_251GEV_EP-SIGMARED, variant: legacy}
        - {dataset: HERA_NC_300GEV_EP-SIGMARED, variant: legacy}
        - {dataset: HERA_NC_318GEV_EP-SIGMARED, variant: legacy}
        # - {dataset: HERA_CC_318GEV_EM-SIGMARED, variant: legacy} # (out-of-sample)
        - {dataset: HERA_CC_318GEV_EP-SIGMARED, variant: legacy}
        - {dataset: HERA_NC_318GEV_EAVG_CHARM-SIGMARED, variant: legacy}
        - {dataset: HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED, variant: legacy}


    theoryid: 40_000_000                          # The theory from which the predictions are drawn
    use_cuts: internal                     # The kinematic cuts to be applied to the data

    closure_test_level: 1
    level_1_seed: 123456
    closure_test_pdf: 250503_pod_basis_40k_underlying_law_40w_pos

    ## NNPDF settings, these are needed by validphys to run report
    closuretest:
        fakedata: true
        filterseed: 123456 # should be the same as level_1_seed
        fakepdf: 250503_pod_basis_40k_underlying_law_40w_pos # should be the same as closure_test_pdf


    #####################
    # Loss function specs
    #####################
    use_fit_t0: true                    # Whether the t0 covariance is used in the chi2 loss
    t0pdfset: 240701-02-rs-nnpdf40-baseline         # The t0 PDF used to build the t0 covariance matrix

    #############
    # Model specs
    #############

    # Weight minimisation settings
    wmin_settings:
        wminpdfset: 250503_pod_basis_40k
        n_basis: 70

    ###################
    # Methodology specs
    ###################

    # Analytic settings
    analytic_settings:
        n_posterior_samples: 1              # Number of posterior samples written to exportgrids, ready for evolution
        full_sample_size: 50000             # Number of samples to be drawn from the posterior
        sampling_seed: 91234                # Random seed used for reproducible sampling


    prior_settings:
        prior_distribution: uniform_parameter_prior
        prior_distribution_specs: 
                {
                "min_val": -2.65,
                "max_val": 2.64,
                }

    actions_:
    - run_analytic_fit                       

Note that the ``prior_settings`` are the same as in a bayesian fit, which you can
find more details about in :ref:`this tutorial <in_running_bayesian>`.

