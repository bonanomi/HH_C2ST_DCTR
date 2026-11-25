"""EDIT paths, datasets, exposure scales and alignment declarations before use.
No paths, 2025 dataset names, or luminosities are guessed.
"""
import c2st_config as old

DATA25 = {
    "Data25 (electron)": [
        "data_e_c",
        "data_e_d",
        "data_e_e",
        "data_e_f",
        "data_e_g",
    ],
    "Data25 (muon)": [
        "data_mu_c",
        "data_mu_d",
        "data_mu_e",
        "data_mu_f",
        "data_mu_g",
    ],
    "Data25 (muon-electron)": [
        "data_muoneg_c",
        "data_muoneg_d",
        "data_muoneg_e",
        "data_muoneg_f",
        "data_muoneg_g",
    ],
}

DATA24 = {
    "Data24 (electron)": [
        "data_e_c",
        "data_e_d",
        "data_e_e",
        "data_e_f",
        "data_e_g",
        "data_e_h",
        "data_e_i",
    ],
    "Data24 (muon)": [
        "data_mu_c",
        "data_mu_d",
        "data_mu_e",
        "data_mu_f",
        "data_mu_g",
        "data_mu_h",
        "data_mu_i",
    ],
    "Data24 (muon-electron)": [
        "data_muoneg_c",
        "data_muoneg_d",
        "data_muoneg_e",
        "data_muoneg_f",
        "data_muoneg_g",
        "data_muoneg_h",
        "data_muoneg_i",
    ],
}

# 1 means raw event yields. For an exposure-normalized rate correction, set
# target scale = L_ref/L25 and base scale = L_ref/L24, using compatible units.
# Equal values cancel in odds. Choose L_ref so weights remain numerically useful.
TARGET_EXPOSURE_SCALE = 1.0
BASE_EXPOSURE_SCALE = 1.0

CONFIG = {
    "description": "Data25 / Data24; transfer hypothesis for MC25",
    "target_processes": DATA25, "base_processes": DATA24,
    "target_source": "2025", "base_source": "2024",
    "sources": {
        "2024": {"format": "cf", "store_root": "/data/dust/user/letzerba/cf_data/hh2bbww/cf_store/hbw_dl/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c24v15/",
                 "reduction_dir": "/data/dust/user/letzerba/public/hh2bbww/data/common_store/hbw_merged/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c24v15/cf.MergeReducedEvents/", "shift": "nominal",
                 "producers": {"dl_ml_inputs": "/data/dust/user/letzerba/cf_data/hh2bbww/cf_store/hbw_dl/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c24v15/cf.ProduceColumns/prod__dl_ml_inputsV0",
                "event_weights": "/data/dust/user/letzerba/cf_data/hh2bbww/cf_store/hbw_dl/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c24v15/cf.ProduceColumns/prod__event_weightsV2"},
                 # Set entries True only after checking positional alignment.
                 "alignment_ok": {d: True for ds in DATA24.values() for d in ds}},
        "2025": {"format": "cf", "store_root": "/data/dust/user/letzerba/cf_data/hh2bbww/cf_store/hbw_dl/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c25v15/",
                 "reduction_dir": "/data/dust/user/letzerba/public/hh2bbww/data/common_store/hbw_merged/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c25v15/cf.MergeReducedEvents/", "shift": "nominal",
                 "producers": {"dl_ml_inputs": "/data/dust/user/letzerba/cf_data/hh2bbww/cf_store/hbw_dl/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c25v15/cf.ProduceColumns/prod__dl_ml_inputsV0",
                "event_weights": "/data/dust/user/letzerba/cf_data/hh2bbww/cf_store/hbw_dl/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c25v15/cf.ProduceColumns/prod__event_weightsV2"},
                 "alignment_ok": {d: True for ds in DATA25.values() for d in ds}},
    },
    "target_weights": {"mode": "unit", "scale": TARGET_EXPOSURE_SCALE},
    "base_weights": {"mode": "unit", "scale": BASE_EXPOSURE_SCALE},
    "subtract_processes": [], "reference_stages": [],
    "region_ids": {"ar": 1, "dycr": 3, "ttcr": 4}, "regions": ["dycr"],
    # Example region union: ["dycr", "ttcr"]. Each row is loaded only once.
    "channel_ids": {"2e": 30, "2mu": 40, "emu": 50},
    "channels": {"2mu": ["2mu"], "2e": ["2e"]},
    # To pool channels into one model: {"combined": ["2mu", "2e"]}.
    "features": list(old.FEATURES), "validation_vars": [], "selections": {},
    "normalization": "yield", "compatibility": False,
    "dctr_model": {"hidden": [50], "batch_normalization": True,
                   "optimizer": "sgd", "learning_rate": .005},
    "closure_model": {"hidden": [50], "batch_normalization": False,
                      "optimizer": "adam", "learning_rate": .001},
}
