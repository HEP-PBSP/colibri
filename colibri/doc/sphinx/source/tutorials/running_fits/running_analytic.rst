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

See :ref:`this section <th_analytic_fits>` for an overview of the theoretical
background of the analytic fit methodology, as well as details on when it might
be appropriate to run an analytic fit.

In this section, we will provide an example that was implemented in the work
presented in Ref. :cite:alp:`Costantini:2025wxp`.


Runcard
-------

The following runcard can be used to run an analytic fit. This has been used to
fit the weight minimisation (``wmin``) model presented in Ref.
:cite:alp:`Costantini:2025wxp`. 

.. code-block:: bash

    meta: 'An example analytic fit using Colibri'

    #######################
    # Data and theory specs
    #######################

    dataset_inputs:
    # DIS data (DEUTERON experiments)
    - {dataset: NMC_NC_NOTFIXED_EM-F2, variant: legacy_dw}
    - {dataset: SLAC_NC_NOTFIXED_P_EM-F2, variant: legacy_dw}
    - {dataset: SLAC_NC_NOTFIXED_D_EM-F2, variant: legacy_dw}
    - {dataset: BCDMS_NC_NOTFIXED_P_EM-F2, variant: legacy_dw}
    - {dataset: BCDMS_NC_NOTFIXED_D_EM-F2, variant: legacy_dw}


    theoryid: 40_000_000                   # The theory from which the predictions are drawn.
    use_cuts: internal                     # The kinematic cuts to be applied to the data.

    closure_test_level: 1
    level_1_seed: 123456
    closure_test_pdf: 250503_pod_basis_40k_underlying_law_40w_pos

    ## NNPDF settings, these are needed by validphys to run report
    closuretest:
    fakedata: true
    filterseed: 123456 # should be the same as level_1_seed
    fakepdf: 250503_pod_basis_40k_underlying_law_40w_pos # should be the same as closure_test_pdf

    positivity:
    posdatasets:
    # Positivity Lagrange Multiplier
    - {dataset: NNPDF_POS_2P24GEV_F2U, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_F2D, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_F2S, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_FLL, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_F2C, maxlambda: 1e6}
    # Positivity of MSbar PDFs
    - {dataset: NNPDF_POS_2P24GEV_XUQ, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_XUB, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_XDQ, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_XDB, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_XSQ, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_XSB, maxlambda: 1e6}
    - {dataset: NNPDF_POS_2P24GEV_XGL, maxlambda: 1e6}


    # Apply a cut to a dataset or process type
    added_filter_rules:
        - process_type: POS_XPDF
        rule: "x > 0.1"

        - process_type: POS_DIS # affects structure functions
        rule: "x > 3e-05"


    positivity_penalty_settings:
        positivity_penalty: true
        alpha: 1e-7           
        lambda_positivity: 100


    integrability_settings:
    integrability: true
    integrability_specs:
        lambda_integrability: 50
        evolution_flavours: [T3, T8]


    ####################
    # Colibri specs
    ####################
    use_fit_t0: True                                # Whether the t0 covariance is used in the chi2 loss.
    t0pdfset: 240701-02-rs-nnpdf40-baseline         # The t0 PDF used to build the t0 covariance matrix.
    
    # Nested Sampling settings
    ns_settings:
    sampler_plot: true # is slow for large number of parameters
    n_posterior_samples: 100
    ReactiveNS_settings:
        vectorized: false
        # ndraw_max: 500
    #    resume: True 
    Run_settings:
        min_num_live_points: 400
        min_ess: 1000
        frac_remain: 0.01
    SliceSampler_settings:
        nsteps: 80

    # Prior settings
    prior_settings:
    prior_distribution: 'prior_from_gauss_posterior'            # The type of prior used in Nested Sampling (model dependent)
    prior_distribution_specs: {"prior_fit": 250510_analytic_L1_}


    #############
    # Model specs
    #############

    # Weight minimisation settings
    wmin_settings:
    wminpdfset: 250503_pod_basis_40k 
    n_basis: 45


    actions_:
    - run_ultranest_fit                       


Note that the relevant options for the initialisation settings are the same as in
a bayesian fit, which you can find more details about in
:ref:`this tutorial <in_running_bayesian>`.

