.. _running_mc_replica:


========================
Monte Carlo replica Fits
========================

Runcard
-------

Here is an example runcard to perform a fit using the Monte Carlo replica
method. Note that the dependence on the model will come from the model-specific
executable. 

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

    # positivity_penalty_settings:
    #   positivity_penalty: false
    #   alpha: 1e-7                           
    #   lambda_positivity: 0                 

    use_fit_t0: True                       # Whether the t0 covariance is used in the chi2 loss.
    t0pdfset: NNPDF40_nnlo_as_01180        # The t0 PDF used to build the t0 covariance matrix.
    

    ###################
    # Methodology specs
    ###################

    # Integrability Settings
    #integrability_settings:
    #  integrability: False

    # Monte Carlo settings
    use_gen_t0: True                       # Whether the t0 covariance is used to generated pseudodata.
    max_epochs: 300                        # The max number of epochs in Monte Carlo training.
    mc_validation_fraction: 0.2            # The fraction of the data used for validation in Monte Carlo training.

    mc_initialiser_settings:               # The initialiser for Monte Carlo training.
        type: uniform
        min_val: 0
        max_val: 3

    actions_:
    - run_monte_carlo_fit

Running the fit
---------------

For example, to perform a Monte Carlo fit for the Les Houches model 
(presented in :ref:`this tutorial <in_les_houches>`), you would run:

.. code-block:: bash

    les_houches_exe monte_carlo_runcard.yml -rep number_of_replicas

Running fits will generate fit folders, the details of which can be found in 
:ref:`this section <mc_fit_folders>`.