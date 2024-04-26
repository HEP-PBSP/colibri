.. _simu_closure_test:

SIMUnet closure test
=====================

Closure tests serve as a mechanism to assess the reliability of fits of PDFs and physical parameters in general, and the robustness of the methodologies involved.

The :math:`\text{SIMUnet}` code extends the capacity of the NNPDF closure test to run contaminated fits and closure tests probing the fit quality of both PDFs and
physical parameters.

:math:`\text{SIMUnet}` can produce BSM contaminated fits by injecting New Physics into the generated pseudodata 
while the fit is done assuming the Standard Model. With this functionality, any user
can test the robustness of any New Physics scenario against being fitted away in a PDF fit.
In this context, :math:`\text{SIMUnet}` can perform closure tests for both PDFs and physical parameters, giving surety of the reliability of the :math:`\text{SIMUnet}`
methodology.

A :math:`\text{SIMUnet}` closure test corresponds to running a BSM contaminated PDF fit and a simultaneous PDF-EFT fit at the same time. 
For details about each process see each tutorial. Here is an example runcard:

.. code-block:: yaml

    # Runcard for contaminated PDF fit with SIMUnet
    #
    ############################################################
    description: "Runcard template for a contaminated PDF fit with BSM physics injected in the data,
    defined as linear combinations of the SMEFT Warsaw basis operators."

    ############################################################
    # frac: training fraction of datapoints for the PDFs
    # QCD: apply QCD K-factors
    # EWK: apply electroweak K-factors
    # simu_fac: fit BSM coefficients using their K-factors in the dataset 
    # use_fixed_predictions:  if set to True it removes the PDF dependence of the dataset
    # sys: systematics treatment (see systypes)

    ############################################################
    dataset_inputs:

    ### 'Standard' datasets ###
    - {dataset: NMCPD_dw_ite, frac: 0.75}
    - {dataset: NMC, frac: 0.75}
    - {dataset: SLACP_dwsh, frac: 0.75}
    - {dataset: HERACOMBCCEP, frac: 0.75}
    - {dataset: HERACOMB_SIGMARED_C, frac: 0.75}
    - {dataset: HERACOMB_SIGMARED_B, frac: 0.75}
    - {dataset: DYE886R_dw_ite, frac: 0.75, cfac: ['QCD']}
    - {dataset: DYE886P, frac: 0.75, cfac: ['QCD']}
    - {dataset: DYE605_dw_ite, frac: 0.75, cfac: ['QCD']}
    - {dataset: DYE906R_dw_ite, frac: 0.75, cfac: ['ACC', 'QCD']}
    - {dataset: CDFZRAP_NEW, frac: 0.75, cfac: ['QCD']}
    - {dataset: D0ZRAP_40, frac: 0.75, cfac: ['QCD']}
    - {dataset: D0WMASY, frac: 0.75, cfac: ['QCD']}
    - {dataset: ATLASWZRAP36PB, frac: 0.75, cfac: ['QCD']}

    ### 'Contaminated' datasets ###
    - {dataset: CMSDY1D12, frac: 0.75, cfac: ['QCD', 'EWK'], contamination: 'EFT_LO', simu_fac: "EFT_LO",}
    - {dataset: CMS_HMDY_13TEV, frac: 0.75, cfac: ['QCD', 'EWK'], contamination: 'EFT_LO', simu_fac: "EFT_LO",}
    - {dataset: ATLASDY2D8TEV, frac: 0.75, cfac: ['QCDEWK'], contamination: 'EFT_LO', simu_fac: "EFT_LO",}
    - {dataset: ATLASZHIGHMASS49FB, frac: 0.75, cfac: ['QCD'], contamination: 'EFT_LO', simu_fac: "EFT_LO",}

    fixed_pdf_fit: False
    # load_weights_from_fit: 221103-jmm-no_top_1000_iterated # If this is uncommented, training starts here.

    simu_parameters:
    - {name: 'Olq3', scale: 10, initialisation: {type: uniform, minval: -1, maxval: 1}}
    - {name: 'OtG', scale: 0.1, initialisation: {type: uniform, minval: -1, maxval: 1}}

    ###########################################################
    # The closure test namespace tells us the settings for the
    # (possible contaminated) closure test.
    ############################################################
    closuretest:
    filterseed: 0 # Random seed to be used in filtering data partitions
    fakedata: true     # true = to use FAKEPDF to generate pseudo-data
    fakepdf: NNPDF40_nnlo_as_01180      # Theory input for pseudo-data
    errorsize: 1.0    # uncertainties rescaling
    fakenoise: true    # true = to add random fluctuations to pseudo-data
    rancutprob: 1.0   # Fraction of data to be included in the fit
    rancutmethod: 0   # Method to select rancutprob data fraction
    rancuttrnval: false # 0(1) to output training(valiation) chi2 in report
    printpdf4gen: false # To print info on PDFs during minimization
    contamination_parameters:
        - name: 'W'
        value: 0.00008
        linear_combination:
            'Olq3': -15.94

        - name: 'Y'
        value: 1 
        linear_combination:
            'Olq1': 1.51606
            'Oed': -6.0606
            'Oeu': 12.1394
            'Olu': 6.0606
            'Old': -3.0394
            'Oqe': 3.0394

    seed: 0
    rngalgo: 0
    ############################################################
    datacuts:
    t0pdfset: NNPDF40_nnlo_as_01180 # PDF set to generate t0 covmat
    q2min: 3.49                        # Q2 minimum
    w2min: 12.5                        # W2 minimum

    ############################################################
    theory:
    theoryid: 200     # database id

    ############################################################
    trvlseed: 475038818
    nnseed: 2394641471
    mcseed: 1831662593
    save: "weights.h5"
    genrep: true      # true = generate MC replicas, false = use real data

    ############################################################


    parameters: # This defines the parameter dictionary that is passed to the Model Trainer
    nodes_per_layer: [25, 20, 8]
    activation_per_layer: [tanh, tanh, linear]
    initializer: glorot_normal
    optimizer:
        clipnorm: 6.073e-6
        learning_rate: 2.621e-3
        optimizer_name: Nadam
    epochs: 30000
    positivity:
        initial: 184.8
        multiplier:
    integrability:
        initial: 184.8
        multiplier:
    stopping_patience: 0.2
    layer_type: dense
    dropout: 0.0
    threshold_chi2: 3.5

    fitting:
    # EVOL(QED) = sng=0,g=1,v=2,v3=3,v8=4,t3=5,t8=6,(pht=7)
    # EVOLS(QED)= sng=0,g=1,v=2,v8=4,t3=4,t8=5,ds=6,(pht=7)
    # FLVR(QED) = g=0, u=1, ubar=2, d=3, dbar=4, s=5, sbar=6, (pht=7)
    fitbasis: EVOL  # EVOL (7), EVOLQED (8), etc.
    basis:
    - {fl: sng, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        1.093, 1.121], largex: [1.486, 3.287]}
    - {fl: g, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.8329, 1.071], largex: [3.084, 6.767]}
    - {fl: v, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.5202, 0.7431], largex: [1.556, 3.639]}
    - {fl: v3, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.1205, 0.4839], largex: [1.736, 3.622]}
    - {fl: v8, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.5864, 0.7987], largex: [1.559, 3.569]}
    - {fl: t3, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        -0.5019, 1.126], largex: [1.754, 3.479]}
    - {fl: t8, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.6305, 0.8806], largex: [1.544, 3.481]}
    - {fl: t15, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        1.087, 1.139], largex: [1.48, 3.365]}

    ############################################################
    positivity:
    posdatasets:
    - {dataset: POSF2U, maxlambda: 1e6}        # Positivity Lagrange Multiplier
    - {dataset: POSF2DW, maxlambda: 1e6}
    - {dataset: POSF2S, maxlambda: 1e6}
    - {dataset: POSFLL, maxlambda: 1e6}
    - {dataset: POSDYU, maxlambda: 1e10}
    - {dataset: POSDYD, maxlambda: 1e10}
    - {dataset: POSDYS, maxlambda: 1e10}
    - {dataset: POSF2C, maxlambda: 1e6}
    - {dataset: POSXUQ, maxlambda: 1e6}        # Positivity of MSbar PDFs
    - {dataset: POSXUB, maxlambda: 1e6}
    - {dataset: POSXDQ, maxlambda: 1e6}
    - {dataset: POSXDB, maxlambda: 1e6}
    - {dataset: POSXSQ, maxlambda: 1e6}
    - {dataset: POSXSB, maxlambda: 1e6}
    - {dataset: POSXGL, maxlambda: 1e6}

    ############################################################
    integrability:
    integdatasets:
    - {dataset: INTEGXT8, maxlambda: 1e2}
    - {dataset: INTEGXT3, maxlambda: 1e2}

    ############################################################
    debug: false
    maxcores: 4


The following section implement the BSM contamination of the data:

.. code-block:: yaml

        ###########################################################
    # The closure test namespace tells us the settings for the
    # (possible contaminated) closure test.
    ############################################################
    closuretest:
    filterseed: 0 # Random seed to be used in filtering data partitions
    fakedata: true     # true = to use FAKEPDF to generate pseudo-data
    fakepdf: NNPDF40_nnlo_as_01180      # Theory input for pseudo-data
    errorsize: 1.0    # uncertainties rescaling
    fakenoise: true    # true = to add random fluctuations to pseudo-data
    rancutprob: 1.0   # Fraction of data to be included in the fit
    rancutmethod: 0   # Method to select rancutprob data fraction
    rancuttrnval: false # 0(1) to output training(valiation) chi2 in report
    printpdf4gen: false # To print info on PDFs during minimization
    contamination_parameters:
        - name: 'W'
        value: 0.00008
        linear_combination:
            'Olq3': -15.94

        - name: 'Y'
        value: 1 
        linear_combination:
            'Olq1': 1.51606
            'Oed': -6.0606
            'Oeu': 12.1394
            'Olu': 6.0606
            'Old': -3.0394
            'Oqe': 3.0394

    seed: 0
    rngalgo: 0
    ############################################################


The following section implement the SMEFT Wilson coefficient fit:

.. code-block:: yaml

        simu_parameters:
    - {name: 'Olq3', scale: 10, initialisation: {type: uniform, minval: -1, maxval: 1}}
    - {name: 'OtG', scale: 0.1, initialisation: {type: uniform, minval: -1, maxval: 1}}


The following section select the datset used in the BSM contamination and for the SMEFT fit:

.. code-block:: yaml

        ### 'Contaminated' datasets ###
    - {dataset: CMSDY1D12, frac: 0.75, cfac: ['QCD', 'EWK'], contamination: 'EFT_LO', simu_fac: "EFT_LO",}
    - {dataset: CMS_HMDY_13TEV, frac: 0.75, cfac: ['QCD', 'EWK'], contamination: 'EFT_LO', simu_fac: "EFT_LO",}
    - {dataset: ATLASDY2D8TEV, frac: 0.75, cfac: ['QCDEWK'], contamination: 'EFT_LO', simu_fac: "EFT_LO",}
    - {dataset: ATLASZHIGHMASS49FB, frac: 0.75, cfac: ['QCD'], contamination: 'EFT_LO', simu_fac: "EFT_LO",}


The fit needs then to be run as a BSM contaminated fit:

.. code-block:: bash

    $ vp-setupfit runcard.yaml
    $ vp-rebuild-data runcard_folder
    $ n3fit runcard.yaml replica_number
    $ evolven3fut runcard_folder replica_number
    $ postfit final_replica_number runcard_folder

Every time a replica is finalized, the output is saved to the `runcard/nnfit/replica_$replica`_
folder, which contains a number of files:

- ``chi2exps.log``: a json log file with the χ² of the training every 100 epochs.
- ``runcard.exportgrid``: a file containing the PDF grid.
- ``runcard.json``: Includes information about the fit (metadata, parameters, times) in json format.
- ``bsm_fac.csv``: Contains the values of the EFT coefficients for this replica.


