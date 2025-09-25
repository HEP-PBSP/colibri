.. _running_hessian:

============
Hessian Fits
============

This section describes how to run Hessian fits with Colibri.

Hessian fits are performed by minimising the :math:`\chi^2` loss defined with respect
to the experimental data. The best-fit parameters are found by minimising the :math:`\chi^2`
using gradient descent. The Hessian matrix is then computed at the minimum and used to define
the covariance matrix of the fit parameters. A parameter called tolerance can be used to inflate
the uncertainties, as is common in global PDF fits. For more details on the Hessian method, see
:cite:ct:`Bailey:2020ooq` and :cite:ct:`Pumplin:2002vw`.

In this tutorial, we will present an example that can be used to fit the
:ref:`Les Houches parametrisation model <lh_theory>`.

Runcard
-------

The following runcard can be used to run a Hessian fit with Colibri.

.. code-block:: bash

    meta: 'An example Hessian fit using Colibri, reduced DIS dataset.'

    #######################
    # Data and theory specs
    #######################

    dataset_inputs:    
    # DIS          
    - {dataset: SLAC_NC_NOTFIXED_P_EM-F2, variant: legacy_dw}
    - {dataset: SLAC_NC_NOTFIXED_D_EM-F2, variant: legacy_dw}
    - {dataset: BCDMS_NC_NOTFIXED_P_EM-F2, variant: legacy_dw}
    - {dataset: BCDMS_NC_NOTFIXED_D_EM-F2, variant: legacy_dw}
    - {dataset: CHORUS_CC_NOTFIXED_PB_NU-SIGMARED, variant: legacy_dw}
    - {dataset: CHORUS_CC_NOTFIXED_PB_NB-SIGMARED, variant: legacy_dw}
    - {dataset: NUTEV_CC_NOTFIXED_FE_NU-SIGMARED, cfac: [MAS], variant: legacy_dw}
    - {dataset: NUTEV_CC_NOTFIXED_FE_NB-SIGMARED, cfac: [MAS], variant: legacy_dw}
    - {dataset: HERA_NC_318GEV_EM-SIGMARED, variant: legacy}
    - {dataset: HERA_NC_225GEV_EP-SIGMARED, variant: legacy}
    - {dataset: HERA_NC_251GEV_EP-SIGMARED, variant: legacy}
    - {dataset: HERA_NC_300GEV_EP-SIGMARED, variant: legacy}
    - {dataset: HERA_NC_318GEV_EP-SIGMARED, variant: legacy}
    - {dataset: HERA_CC_318GEV_EM-SIGMARED, variant: legacy}
    - {dataset: HERA_CC_318GEV_EP-SIGMARED, variant: legacy}
    - {dataset: HERA_NC_318GEV_EAVG_CHARM-SIGMARED, variant: legacy}
    - {dataset: HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED, variant: legacy}
    - {dataset: NMC_NC_NOTFIXED_EM-F2, variant: legacy_dw}
    - {dataset: NMC_NC_NOTFIXED_P_EM-SIGMARED, variant: legacy}



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
    hessian_settings:
        tolerance: 1.0
        # n_samples: 2 # only used if ErrorType: replicas
        iter_init: 1
        ErrorType: hessian
        n_eigvec: 20 # Number of eigenvectors to use. If not set, defaults to 20.
        # rng_seed: 123456


    optimizer_settings:
        optimizer: adam # Any of the optax optimizers can be used.
        # clipnorm: 1.0 # Gradient clipping norm value. Set to null to disable or omit.
        optimizer_hyperparams:
            learning_rate: 0.001
            # any hyperparameters specific to the chosen optimizer can be set here

    # Training settings
    use_gen_t0: True             # Whether the t0 covariance is used to generated pseudodata.
    max_epochs: 30000
    patience: 30000

    param_initialiser_settings:               
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

    actions_:
    - run_hessian_fit


Note that the Hessian fit uses the same ``param_initialiser_settings`` as a
Monte Carlo fit, and so any of the initialisation options discussed in the
:ref:`Monte Carlo fit tutorial <running_mc_replica>` can be used for these
settings (i.e. gaussian initialisation, global bounds for all parameters ...).

``hessian_settings``
^^^^^^^^^^^^^^^^^^^^
* ``ErrorType``: You can chose between ``hessian`` and ``replicas``. In the case of ``hessian``, 
    the Hessian matrix is diagonalised and the eigenvectors are used to define the uncertainties.
    In the case of ``replicas``, a set of parameter replicas are generated by sampling from a multivariate
    Gaussian defined by the inverse of the Hessian matrix. The number of replicas is set by ``n_samples``.
* ``tolerance``: This parameter inflates the uncertainties obtained from the Hessian matrix by 
    multiplying the covariance matrix by T^2.
* ``n_samples``: Number of replicas to sample from approximated gaussian. 
    This is only used if ``ErrorType`` is set to ``replicas``.
* ``n_eigvec``: Number of eigenvectors to use. If not set, defaults to 20. Only used if ``ErrorType`` is set to ``hessian``.
* ``iter_init``: Set this to however many different starting points to try to initialise from. The one with the lowest :math:`\chi^2` will be taken. This is done in order to avoid local minima.
* ``rng_seed``: Controls the random number generator seed for reproducibility.

Running the fit
---------------

To run a Hessian fit for the Les Houches model 
(presented in :ref:`this tutorial <in_les_houches>`), you would run (in case you have not done so already):

.. code-block:: bash

    pip install -e .

from the ``examples/les_houches_example`` directory. This enables the
model-specific executable.

Then, you can use the above runcard with the following command:

.. code-block:: bash

    les_houches_exe hessian_runcard.yml 

Finally, you can evolve the fit, as described in :ref:`this section <evolution_script>`.
If the fit has been run with ``ErrorType: hessian``, evolution will need to be performed with the flag
``--hessian_fit True``.