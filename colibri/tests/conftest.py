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


TEST_FULL_HAD_DATASET = {
    "dataset_inputs": [
        # Hadronic
        {"dataset": "DYE866_Z0_800GEV_DW_RATIO_PDXSECRATIO", "variant": "legacy"},
        {"dataset": "DYE866_Z0_800GEV_PXSEC", "variant": "legacy"},
        {"dataset": "DYE605_Z0_38P8GEV_DW_PXSEC", "variant": "legacy"},
        {
            "dataset": "DYE906_Z0_120GEV_DW_PDXSECRATIO",
            "variant": "legacy",
            "cfac": ["ACC"],
        },
        {"dataset": "CDF_Z0_1P96TEV_ZRAP", "variant": "legacy"},
        {"dataset": "D0_Z0_1P96TEV_ZRAP", "variant": "legacy"},
        {"dataset": "D0_WPWM_1P96TEV_ASY", "variant": "legacy"},
        {"dataset": "ATLAS_DY_7TEV_36PB_ETA", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_49FB_HIMASS", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_LOMASS_M", "variant": "legacy"},
        {"dataset": "ATLAS_DY_7TEV_46FB_CC", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_7TEV_46FB_CF-Y", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_8TEV_HIMASS_M-Y", "variant": "legacy"},
        {"dataset": "ATLAS_Z0_8TEV_LOWMASS_M-Y", "variant": "legacy"},
        {"dataset": "ATLAS_DY_13TEV_TOT", "variant": "legacy", "cfac": ["NRM"]},
        {"dataset": "ATLAS_WJ_JET_8TEV_WP-PT", "variant": "legacy"},
        {"dataset": "ATLAS_WJ_JET_8TEV_WM-PT", "variant": "legacy"},
        {"dataset": "ATLAS_Z0J_8TEV_PT-M", "variant": "legacy"},
        {"dataset": "ATLAS_Z0J_8TEV_PT-Y", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_7TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_13TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_PH_13TEV_XSEC", "variant": "legacy", "cfac": ["EWK"]},
        {"dataset": "ATLAS_SINGLETOP_7TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_13TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_7TEV_T-Y-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_7TEV_TBAR-Y-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_8TEV_T-RAP-NORM", "variant": "legacy"},
        {"dataset": "ATLAS_SINGLETOP_8TEV_TBAR-RAP-NORM", "variant": "legacy"},
        {"dataset": "CMS_WPWM_7TEV_ELECTRON_ASY"},
        {"dataset": "CMS_WPWM_7TEV_MUON_ASY", "variant": "legacy"},
        {"dataset": "CMS_Z0_7TEV_DIMUON_2D"},
        {"dataset": "CMS_WPWM_8TEV_MUON_Y", "variant": "legacy"},
        {"dataset": "CMS_Z0J_8TEV_PT-Y", "cfac": ["NRM"], "variant": "legacy"},
        {"dataset": "CMS_TTBAR_7TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_5TEV_TOT_X-SEC", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_2L_DIF_YT", "variant": "legacy"},
        {"dataset": "CMS_TTBAR_13TEV_LJ_2016_DIF_YTTBAR", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_7TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_8TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "CMS_SINGLETOP_13TEV_TCHANNEL-XSEC", "variant": "legacy"},
        {"dataset": "LHCB_Z0_7TEV_DIELECTRON_Y"},
        {"dataset": "LHCB_Z0_8TEV_DIELECTRON_Y"},
        {"dataset": "LHCB_DY_7TEV_MUON_Y", "cfac": ["NRM"]},
        {"dataset": "LHCB_DY_8TEV_MUON_Y", "cfac": ["NRM"]},
        {"dataset": "LHCB_Z0_13TEV_DIMUON-Y"},
        {"dataset": "LHCB_Z0_13TEV_DIELECTRON-Y"},
    ],
    "theoryid": 700,
    "use_cuts": "internal",
}
