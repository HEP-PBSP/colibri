.. _contfit:

Contaminated fits
=================

This is a basic tutorial to perform PDF fits contaminated with BSM physics with SIMUnet.
The procedure consists of 3 steps: 

1. `Preparing a fit runcard <#preparing-a-fit-runcard>`_
2. `Running the fitting code <#running-the-fitting-code>`_
3. `Uploading and analysing the results of the fit <#upload-and-analyse-the-fit>`_

.. _preparing-a-fit-runcard:

1. Preparing a fit runcard
--------------------------

The runcard is written in YAML. The runcard is the unique identifier of a fit
and contains all required information to perform a fit, which includes the
experimental data, the theory setup and the fitting setup.

We begin by showing the user an example of a complete runcard. We will go
into the details of each part  later. Here is a complete SIMUnet runcard:

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
    - {dataset: CMSDY1D12, frac: 0.75, cfac: ['QCD', 'EWK'], contamination: 'EFT_LO'}
    - {dataset: CMS_HMDY_13TEV, frac: 0.75, cfac: ['QCD', 'EWK'], contamination: 'EFT_LO'}
    - {dataset: ATLASDY2D8TEV, frac: 0.75, cfac: ['QCDEWK'], contamination: 'EFT_LO'}
    - {dataset: ATLASZHIGHMASS49FB, frac: 0.75, cfac: ['QCD'], contamination: 'EFT_LO'}

    fixed_pdf_fit: False
    # load_weights_from_fit: 221103-jmm-no_top_1000_iterated # If this is uncommented, training starts here.

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

The structure of the runcard is similar to the one that is used in the NNPDF methodology.
So, in this tutorial we will mostly adress the new syntax and features of SIMUnet. 

We begin by looking at the following section of the runcard:

.. code-block:: yaml

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
    - {dataset: CMSDY1D12, frac: 0.75, cfac: ['QCD', 'EWK'], contamination: 'EFT_LO'}
    - {dataset: CMS_HMDY_13TEV, frac: 0.75, cfac: ['QCD', 'EWK'], contamination: 'EFT_LO'}
    - {dataset: ATLASDY2D8TEV, frac: 0.75, cfac: ['QCDEWK'], contamination: 'EFT_LO'}
    - {dataset: ATLASZHIGHMASS49FB, frac: 0.75, cfac: ['QCD'], contamination: 'EFT_LO'}

The ``dataset_inputs`` key contains the datasets that will be used to peform the PDF fit. 
The ``'Standard' datasets`` are included in the same way as in a NNPDF fit. The ``'Contaminated' datasets`` 
are datasets that are contaminated with BSM physics. The contamination is activated by the ``contamination_parameters`` key. 
The actual BSM contamination is defined in the next section of the runcard:

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

The ``contamination_parameters`` key defines the BSM parameters that will be used to contaminate the datasets. In this case the ``W`` 
parameter encodes the 4-fermion interaction induced by a heavy W' boson, while the ``Y`` parameter encodes the 4-fermion interaction 
induced by a heavy Z' boson. In practice one needs to define the linear combination of the SMEFT Warsaw basis operators that will be 
describing the BSM physics.

.. _running-the-fitting-code:

2. Running the fitting code
---------------------------

After preparing a SIMUnet runcard ``runcard.yml``, we are now ready to run a fit. The pipeline
is similar to the NNPDF framework but some additional features can be included. In practice a contaminated 
fit can be run where the runcard is located by running the following command:

.. code-block:: bash

    $ vp-setupfit runcard.yaml
    $ vp-rebuild-data runcard_folder
    $ n3fit runcard.yaml replica_number
    $ evolven3fut runcard_folder replica_number
    $ postfit final_replica_number runcard_folder

Here is a breakdown of what each command does:

1. Preparing the fit: ``vp-setupfit runcard.yml``
    This command will generate a folder with the same name as the runcard (minus the file extension) in the
    current directory, which will contain a copy of the original YAML runcard.
    The required resources (such as the theory and t0 PDF set) will be
    downloaded automatically. Alternatively they can be obtained with the
    ``vp-get`` tool.

    .. note::
       This step is not strictly necessary when producing a standard fit with
       ``n3fit`` but it is required by :ref:`validphys <vp-index>`
       and it should therefore always be done. Note that :ref:`vp-upload <upload-fit>`
       will fail unless this step has been followed. If necessary, this step can
       be done after the fit has been run.

2. Creating the BSM pseudodata: ``vp-rebuild-data runcard_folder``
    This command will take the generated folder as an argument and will create the BSM contaminated datasets, applying the BSM c-factors
    defined in the runcard to the experimental commondata. The contaminated data is stored in the runcard fit folder
    and will be used for the rest of the fit.

3. Running the fit: ``n3fit runcard.yaml replica``
    The ``n3fit`` program takes a ``runcard.yml`` as input and a replica number, e.g.
    ``n3fit runcard.yml replica`` where ``replica`` goes from 1-n where n is the
    maximum number of desired replicas. Note that if you desire, for example, a 100
    replica fit you should launch more than 100 replicas (e.g. 130) because not
    all of the replicas will pass the checks in ``postfit``.


4. Evolving the replicas' scale: ``evolven3fit runcard_folder replica``
    Wait until you have fit results. Then
    run the ``evolven3fit`` program once to evolve all replicas using DGLAP. Remember
    to use the total number of replicas run (130 in the
    above example), rather than the number you desire in the final fit.


5. Selecting the replicas: ``postfit final_replica_number runcard_folder``
    Wait until you have results, then run the command to finalize the PDF set by applying post selection criteria.
    This will produce a set of ``final_replica_number + 1`` replicas. This time the
    number of replicas should be that which you desire in the final fit (100 in the
    above example). Note that the
    standard behaviour of ``postfit`` can be modified by using various flags.
    More information can be found at `Processing a fit <postfit>`_.

Output of the fit
-----------------

The output of the fit is stored in the ``runcard_folder``. It is identical to a normal NNPDF output.


3. Uploading the fit
----------------------------------

Once the fit is complete, the next steps involve uploading the results. This is particularly useful
if, for example, you ran the fit on a cluster and want to make it avaiable to collaborators or download it
from a different machine. You can upload the fit by using ``vp-upload runcard_folder`` and then fetch it
with ``vp-get fit fit_name``. Note that, to upload the fit, appropriate credentials are required.