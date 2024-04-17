"""
Module
"""

TEST_DATASET = {
    "dataset_input": {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"},
    "theoryid": 700,
    "use_cuts": "internal",
}

TEST_DATASET_HAD = {
    "dataset_input": {"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"},
    "theoryid": 700,
    "use_cuts": "internal",
}

TEST_DATASETS = {
    "dataset_inputs": [
        {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"}
    ],
    "theoryid": 700,
    "use_cuts": "internal",
}

T0_PDFSET = {"t0pdfset": "NNPDF40_nnlo_as_01180"}

CLOSURE_TEST_PDFSET = {"closure_test_pdf": "NNPDF40_nnlo_as_01180"}

TRVAL_INDEX = {"trval_index": 1}
REPLICA_INDEX = {"replica_index": 1}

TEST_PDFSET = "NNPDF40_nnlo_as_01180"

PSEUDODATA_SEED = 123456


TEST_FULL_DIS_DATASET = {
    "dataset_inputs": [
        {"dataset": "NMC_NC_NOTFIXED_P_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "SLAC_NC_NOTFIXED_P_DW_EM-F2", "variant": "legacy"},
        {"dataset": "SLAC_NC_NOTFIXED_D_DW_EM-F2", "variant": "legacy"},
        {"dataset": "BCDMS_NC_NOTFIXED_P_DW_EM-F2", "variant": "legacy"},
        {"dataset": "BCDMS_NC_NOTFIXED_D_DW_EM-F2", "variant": "legacy"},
        {"dataset": "CHORUS_CC_NOTFIXED_PB_DW_NU-SIGMARED", "variant": "legacy"},
        {"dataset": "CHORUS_CC_NOTFIXED_PB_DW_NB-SIGMARED", "variant": "legacy"},
        {
            "dataset": "NUTEV_CC_NOTFIXED_FE_DW_NU-SIGMARED",
            "variant": "legacy",
            "cfac": ["MAS"],
        },
        {
            "dataset": "NUTEV_CC_NOTFIXED_FE_DW_NB-SIGMARED",
            "variant": "legacy",
            "cfac": ["MAS"],
        },
        {"dataset": "HERA_NC_318GEV_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_251GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_300GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_225GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_CC_318GEV_EP-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_CC_318GEV_EM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED", "variant": "legacy"},
        {"dataset": "HERA_NC_318GEV_EAVG_CHARM-SIGMARED", "variant": "legacy"},
    ],
    "theoryid": 700,
    "use_cuts": "internal",
}
