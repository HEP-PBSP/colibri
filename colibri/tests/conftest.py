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
