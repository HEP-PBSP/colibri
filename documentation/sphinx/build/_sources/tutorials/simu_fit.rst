.. _simufit:

The :math:`\text{SIMUnet}` methodology can perform simultaneous fits of PDF and EFT coefficients. 

Simultaneous fits
====================

This is a basic tutorial to perform simultaneous PDF-EFT fits with :math:`\text{SIMUnet}`.
The procedure consists of 3 steps: 

1. `Preparing the runcard <#preparing-the-runcard>`_
2. `Running the fitting code <#running-the-fitting-code>`_
3. `Uploading the fit <#upload-fit>`_
4. `Analysing the fit <#analyse-fit>`_

.. _preparing-the-runcard:

1. Preparing the runcard
--------------------------

The structure of :math:`\text{SIMUnet}` runcards is similar to the ones used in NNPDF, and more details can
be found on their `website <https://docs.nnpdf.science/n3fit/runcard_detailed.html>`_.

The runcard, a critical component of the fitting process, is formatted using 
YAML (YAML Ain't Markup Language), a human-readable data serialization 
standard. As the cornerstone of each fit, the runcard uniquely identifies it 
and encapsulates all necessary parameters. This includes the selection of 
experimental data, the configuration of the theoretical framework, in the SM and beyond, and the 
specifics of the fitting procedure. For those new to YAML, a comprehensive 
guide and documentation can be found at the `official YAML website 
<https://yaml.org/>`_.

We begin by showing the user an example of a complete runcard. We will go
into the details of each part  later. Here is a complete :math:`\text{SIMUnet}` runcard:

.. code-block:: yaml

    # Runcard for SIMUnet
    #
    ############################################################
    description: "Example runcard. This one performs a simultaenous PDF-EFT fit using data from different sectors."
    ############################################################
    # frac: training fraction of datapoints for the PDFs
    # QCD: apply QCD K-factors
    # EWK: apply electroweak K-factors
    # simu_fac: fit BSM coefficients using their K-factors in the dataset
    # use_fixed_predictions:  if set to True it removes the PDF dependence of the dataset
    # sys: systematics treatment (see systypes)

    dataset_inputs:
    # # DIS
    - {dataset: HERACOMBNCEP460, frac: 0.75}
    # # Drell - Yan
    - {dataset: CMSDY1D12, cfac: ['QCD', 'EWK']}
    # # ttbar
    - {dataset: ATLASTTBARTOT7TEV, cfac: [QCD], simu_fac: "EFT_NLO"}
    # # ttbar AC
    - {dataset: ATLAS_TTBAR_8TEV_ASY, cfac: [QCD], simu_fac: "EFT_NLO"}
    # # TTZ
    - {dataset: ATLAS_TTBARZ_8TEV_TOTAL, simu_fac: "EFT_LO"}
    # # TTW
    - {dataset: ATLAS_TTBARW_8TEV_TOTAL, simu_fac: "EFT_LO"}
    # # single top
    - {dataset: ATLAS_SINGLETOP_TCH_7TEV_T, cfac: [QCD], simu_fac: "EFT_NLO"}
    # # tW
    - {dataset: ATLAS_SINGLETOPW_8TEV_TOTAL, simu_fac: "EFT_NLO"}
    # # W helicity
    - {dataset: ATLAS_WHEL_13TEV, simu_fac: "EFT_NLO", use_fixed_predictions: True}
    # # tt gamma
    - {dataset: ATLAS_TTBARGAMMA_8TEV_TOTAL, simu_fac: "EFT_LO", use_fixed_predictions: True}
    # # tZ
    - {dataset: ATLAS_SINGLETOPZ_13TEV_TOTAL, simu_fac: "EFT_LO", use_fixed_predictions: True}
    # # EWPO
    - {dataset: LEP_ZDATA, simu_fac: "EFT_LO", use_fixed_predictions: True}
    #  Higgs
    - {dataset: ATLAS_CMS_SSINC_RUNI, simu_fac: "EFT_NLO", use_fixed_predictions: True}
    # Diboson
    - {dataset: LEP_EEWW_182GEV, simu_fac: "EFT_LO", use_fixed_predictions: True}


    ############################################################
    # Uncomment to perform fixed-PDF fit
    #fixed_pdf_fit: True
    #load_weights_from_fit: 221103-jmm-no_top_1000_iterated

    ############################################################
    # Analytic initialisation features
    analytic_initialisation_pdf: 221103-jmm-no_top_1000_iterated
    analytic_check: False
    automatic_scale_choice: False

    ############################################################
    simu_parameters:
    # Dipoles
    - {name: "OtG", scale: 0.01, initialisation: {type: uniform, minval: -10, maxval: 10} }
    # Quark Currents
    - {name: "Opt", scale: 0.1, initialisation: {type: gaussian, mean: 0, std_dev: 1} }
    # Lepton currents
    - {name: "O3pl", scale: 1.0, initialisation: {type: constant, value: 0} }
    # linear combination
    - name: 'Y'
      linear_combination:
        'Olq1 ': 1.51606
        'Oed ': -6.0606
        'Oeu ': 12.1394
        'Olu ': 6.0606
        'Old ': -3.0394
        'Oqe ': 3.0394
      scale: 1.0
      initialisation: {type: uniform , minval: -1, maxval: 1}

    ############################################################
    datacuts:
      t0pdfset: 221103-jmm-no_top_1000_iterated # PDF set to generate t0 covmat
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
      - {dataset: POSF2U, maxlambda: 1e6}

    ############################################################
    integrability:
      integdatasets:
      - {dataset: INTEGXT8, maxlambda: 1e2}

    ############################################################
    debug: false
    maxcores: 4

As we said, the structure of the runcard is similar to the one that is used in the NNPDF methodology.
So, in this tutorial we will mostly adress the new features and syntax of :math:`\text{SIMUnet}`. 

We begin by looking at the following section of the runcard:

.. code-block:: yaml

    # Runcard for SIMUnet
    #
    ############################################################
    description: "Example runcard. This one performs a simultaenous PDF-EFT fit using data from different sectors."

    ############################################################
    # frac: training fraction of datapoints for the PDFs
    # QCD: apply QCD K-factors
    # EWK: apply electroweak K-factors
    # simu_fac: fit BSM coefficients using their K-factors in the dataset 
    # use_fixed_predictions:  if set to True it removes the PDF dependence of the dataset
    # sys: systematics treatment (see systypes)

It contains the description of the runcard and some short comments about new keys
of :math:`\text{SIMUnet}`. The user should always provide a useful ``description`` of the runcard as
it will appear when running analyses and can provide information to other people studying the fit.

Now we consider the following fraction of the runcard:

.. code-block:: yaml

    dataset_inputs:
    # # DIS
    - {dataset: HERACOMBNCEP460, frac: 0.75}
    # # Drell - Yan
    - {dataset: CMSDY1D12, cfac: ['QCD', 'EWK']}
    # # ttbar
    - {dataset: ATLASTTBARTOT7TEV, cfac: [QCD], simu_fac: "EFT_NLO"}
    # # ttbar AC
    - {dataset: ATLAS_TTBAR_8TEV_ASY, cfac: [QCD], simu_fac: "EFT_NLO"}
    # # TTZ
    - {dataset: ATLAS_TTBARZ_8TEV_TOTAL, simu_fac: "EFT_LO"}
    # # TTW
    - {dataset: ATLAS_TTBARW_8TEV_TOTAL, simu_fac: "EFT_LO"}
    # # single top
    - {dataset: ATLAS_SINGLETOP_TCH_7TEV_T, cfac: [QCD], simu_fac: "EFT_NLO"}
    # # tW
    - {dataset: ATLAS_SINGLETOPW_8TEV_TOTAL, simu_fac: "EFT_NLO"}
    # # W helicity
    - {dataset: ATLAS_WHEL_13TEV, simu_fac: "EFT_NLO", use_fixed_predictions: True}
    # # tt gamma
    - {dataset: ATLAS_TTBARGAMMA_8TEV_TOTAL, simu_fac: "EFT_LO", use_fixed_predictions: True}
    # # tZ
    - {dataset: ATLAS_SINGLETOPZ_13TEV_TOTAL, simu_fac: "EFT_LO", use_fixed_predictions: True}
    # # EWPO
    - {dataset: LEP_ZDATA, simu_fac: "EFT_LO", use_fixed_predictions: True}
    #  Higgs
    - {dataset: ATLAS_CMS_SSINC_RUNI, simu_fac: "EFT_NLO", use_fixed_predictions: True}
    # Diboson
    - {dataset: LEP_EEWW_182GEV, simu_fac: "EFT_LO", use_fixed_predictions: True}

The ``dataset_inputs`` key contains the datasets that will be used to peform the
simultaneous PDF-EFT fit. The first two datasets, ``HERACOMBNCEP460`` and
``CMSDY1D12``, are included in the same way as in a NNPDF fit, and are
used only to fit the PDF parameters. All the other datasets have the key ``simu_fac`` set to either
``EFT_LO`` or ``EFT_NLO``. This means that :math:`\text{SIMUnet}` will use those datasets to fit
EFT coefficients at the desired accuracy, LO or NLO. The fit requires EFT K-factors for all
the datasets that have the ``simu_fac`` key. Additionally, some datasets have the key ``use_fixed_predictions``
set to ``True``. This means that the PDF dependence is removed from this dataset and, effectively,
the dataset becomes PDF-independent.

   .. note::
      This tutorial describes how to perform a simultaenous PDF-EFT. So, as an aside, we will briefly comment this part of the runcard (which,
      obviously, becomes relevant only if uncommented):

      .. code-block:: yaml

          #fixed_pdf_fit: True # If this is uncommented the PDFs are fixed during the fit and only the EFT coefficients are optimised
          #load_weights_from_fit: 221103-jmm-no_top_1000_iterated # If the line above is uncommented, the weights of the PDF are loaded from here

      These keys, if uncommented, allow the user to perform a fixed-PDF fit. This means that only
      the EFT coefficients are found during the optimisation. If ``fixed_pdf_fit: True``, the PDF weights
      are loaded from the fit ``221103-jmm-no_top_1000_iterated``.

We now check:

.. code-block:: yaml

    # Analytic initialisation features
    analytic_initialisation_pdf: 221103-jmm-no_top_1000_iterated
    analytic_check: False
    automatic_scale_choice: False

Each EFT coefficient has a ``scale`` parameter that quantifies its effective learning rate during the training. The
The optimal scale is usually determined a posteriori after performing a first iteration of the fit, and it should be of
the size of the EFT coefficient's best-fit value. However, :math:`\text{SIMUnet}` can also assist by proposing an automatic
scale choice. The way to understand the code above is by first discussing the ``automatic_scale_choice`` feature. If set to ``True``,
the code will first compute the analytical solution of the EFT coefficient by minimising the loss function. This minimum obviously depends
on the theory prediction, which is calculated using the PDF set given in ``analytic_initialisation_pdf``. The key ``analytic_check``, is set to ``True``,
prints the value of the analytic solution of the EFT coefficient found using a fixed-PDF setting. In this particula runcard, the analytic initialisation
features are simply not used.

We move on to this part of the runcard:

.. code-block:: yaml

    simu_parameters:
    # Dipoles
    - {name: "OtG", scale: 0.01, initialisation: {type: uniform, minval: -10, maxval: 10} }
    # Quark Currents
    - {name: "Opt", scale: 0.1, initialisation: {type: gaussian, mean: 0, std_dev: 1} }
    # Lepton currents
    - {name: "O3pl", scale: 1.0, initialisation: {type: constant, value: 0} }
    # linear combination
    - name: 'Y'
      linear_combination:
        'Olq1 ': 1.51606
        'Oed ': -6.0606
        'Oeu ': 12.1394
        'Olu ': 6.0606
        'Old ': -3.0394
        'Oqe ': 3.0394
      scale: 1.0
      initialisation: { type: uniform , minval: -1, maxval: 1}

This block contains the EFT coefficients that are going to be fitted. Each one
of them has a key ``name``. The name usually resembles the notation of the Warsaw
basis, and they have to match the name of the EFT operators that were used
to produce the K-factors of the datasets in the previous section. 

Also, each EFT coefficient has a ``scale``. This scale is used to modify the size of the learning
rate for this coefficient within the :math:`\text{SIMUnet}` framework. The size of the ``scale``
for an EFT coefficient can speed up the training and, in the case, of a big K-factor,
the convergence to the minimum of the loss function without going over it.

There are several types of initialisation of the EFT coefficients. The ``initialization`` key provides
SIMUnet with instructions for setting parameter values at the start of the training. There are three ways
of doing this:

- When ``uniform`` is chosen, it initializes the parameter value to a random number within the range specified by the ``minval`` and ``maxval`` keys, which need to be set in advance.

- When ``gaussian`` is selected, it sets the parameter's initial value based on a Gaussian distribution using the provided ``mean`` and ``std_dev`` keys to define its mean and standard deviation.

- When ``constant`` is used, it assigns the parameter's initial value directly to the value specified by the key, eliminating the element of randomness from this step.

At this points, we can now run the fitting code.

.. _run-n3fit-fit:

2. Running the fitting code
---------------------------

After preparing a :math:`\text{SIMUnet}` runcard ``runcard.yml``, we are now ready to run a fit. The pipeline
is similar to the NNPDF framework, and the details can be found `here <https://docs.nnpdf.science/tutorials/run-fit.html>`_.
The procedure can be summarised as follows: 

1. Start the fit with ``vp-setupfit runcard.yml``, which will create a dedicated
   directory and fetch necessary resources. Alternatively, use ``vp-get`` for
   manual resource acquisition of a fit.

2. Launch the fit using ``n3fit runcard.yml replica``, specifying the replica number.
   Initiate more replicas than needed to account for potential postfit rejections.

3. Once fits are complete, use ``evolven3fit runcard_folder number_of_replicas`` to evolve replicas
   using the DGLAP. Use the actual number of replicas with which you started.

   .. note::
      For fixed-PDF fits, ``vp-fakeevolve`` can replace ``evolven3fit``. This is much faster to run,
      as the PDFs that are load in a fixed-PDF fit have already been evolved!

4. Finalise with ``postfit number_of_replicas runcard_folder``, which filters replicas
   to yield the final PDF and EFT set. The number specified should match your desired final count.

Output of the fit
-----------------
As in NNPDF, every time a replica is finalised, the output is saved to the `runcard/nnfit/replica_$replica`_
folder, which contains these files:

- ``chi2exps.log``: a log file with the χ² of the training.
- ``runcard.exportgrid``: the PDF grid.
- ``runcard.json``: a json file with the information of the fit.

Additinally, in :math:`\text{SIMUnet}` you will find this file:

- ``bsm_fac.csv``: file with the values of the EFT coefficients for this replica.

Once the fit is complete, the next steps involve uploading and analysing the results.

.. _upload-fit:

3. Uploading the fit
----------------------------------

Once the fit is complete, the next steps involve uploading the results. This is particularly useful
if, for example, you ran the fit on a cluster and want to make it avaiable to collaborators or download it
from a different machine. You can upload the fit by using ``vp-upload runcard_folder`` and then fetch it
with ``vp-get fit fit_name``.


.. _analyse-fit:

4. Analising the fit
--------------------

:math:`\text{SIMUnet}` has different functionalities that allow the user to analyse their results. 
The code inherits functions from `NNPDF <https://docs.nnpdf.science/>`_ and, on their website, the user can find further details and instructions
on how to `analyse results <https://docs.nnpdf.science/tutorials/index.html#analysing-results>`_ when it comes to the PDFs.

:math:`\text{SIMUnet}`, additionally, contains a complete set of functions that allow the user to analyse the EFT space, and the interplay between the PDFs
and the EFT coefficients. The complete documentation can be found on the Functions documentation tab.
