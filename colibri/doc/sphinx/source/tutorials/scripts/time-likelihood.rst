.. _time_likelihood:

=========================
Timing the log-Likelihood
=========================

In this tutorial, we show how to time the log-likelihood for different
batch sizes of the vectorised likelihood. This can help you understand
what the computational cost is for the computation of the log-likelihood
for your machine and for a given model.

The following runcard can be used to time the log-likelihood for the
:ref:`Les Houches parametrisation model <lh_theory>`.


.. code-block:: bash

    meta: 'An example script to time the log-likelihood using Colibri.'

    #######################
    # Data and theory specs
    #######################

    dataset_inputs:
    - {dataset: NMC_NC_NOTFIXED_EM-F2, frac: 0.75, variant: legacy_dw}
    # - {dataset: NMC_NC_NOTFIXED_P_EM-SIGMARED, frac: 0.75, variant: legacy}
    # - {dataset: SLAC_NC_NOTFIXED_P_EM-F2, frac: 0.75, variant: legacy_dw}
    # - {dataset: SLAC_NC_NOTFIXED_D_EM-F2, frac: 0.75, variant: legacy_dw}
    # - {dataset: BCDMS_NC_NOTFIXED_P_EM-F2, frac: 0.75, variant: legacy_dw}
    # - {dataset: BCDMS_NC_NOTFIXED_D_EM-F2, frac: 0.75, variant: legacy_dw}


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



    # needed for time_log_likelihood
    param_initialiser_settings:               # The initialiser for Monte Carlo training.
        type: uniform
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
    
    batch_sample_sizes: [1, 10, 33, 100, 1000, 5000]


    actions_:
    - time_log_likelihood

The ``batch_sample_sizes`` is optional. If it is not specified, default values of
``[1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]`` will be used. These are
the batch sizes for which the likelihood will be timed.

You can run the script with:

.. code-block:: bash

    les_houches_exe time_likelihood.yaml

where ``time_likelihood.yaml`` is the name of your runcard. Note that we are using 
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
