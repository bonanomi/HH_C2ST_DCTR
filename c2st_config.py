from pathlib import Path

STORE_ROOT = Path('/data/dust/user/letzerba/cf_data/hh2bbww/cf_store/hbw_dl/calib__ak4V0__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c24v15/')
REDUCTION_DIR = Path('/data/dust/user/letzerba/public/hh2bbww/data/common_store/hbw_merged/calib__ak4V0__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c24v15/cf.MergeReducedEvents')
SHIFT = 'nominal'

ARTIFACT_DIR = Path('c2st_artifacts')
PLOT_DIR = Path('c2st_validation_plots')

MC_PROCESSES = {
    'DY (ee)': {'datasets': [
        'dy_ee_m10to50_amcatnlo', 'dy_ee_m50toinf_amcatnlo',
        'dy_ee_m50toinf_0j_amcatnlo', 'dy_ee_m50toinf_1j_amcatnlo', 'dy_ee_m50toinf_2j_amcatnlo'], 'is_dy': True},
    'DY (mumu)': {'datasets': [
        'dy_mumu_m10to50_amcatnlo', 'dy_mumu_m50toinf_amcatnlo',
        'dy_mumu_m50toinf_0j_amcatnlo', 'dy_mumu_m50toinf_1j_amcatnlo', 'dy_mumu_m50toinf_2j_amcatnlo'], 'is_dy': True},
    'DY (tautau)': {'datasets': [
        'dy_tautau_m10to50_amcatnlo', 'dy_tautau_m50toinf_amcatnlo',
        'dy_tautau_m50toinf_0j_amcatnlo', 'dy_tautau_m50toinf_1j_amcatnlo', 'dy_tautau_m50toinf_2j_amcatnlo'], 'is_dy': True},
    'ttbar DL': {'datasets': ['tt_dl_powheg'], 'is_dy': False},
    'ttbb': {'datasets': ['ttbb_dl_powheg'], 'is_dy': False},
}
DATA_PROCESSES = {
    'Data (mumu)': [f'data_mu_{era}' for era in 'cdefghi'],
    'Data (ee)': [f'data_e_{era}' for era in 'cdefghi'],
}

# Kept from the checked setup. Re-run dyvr_lib.check_alignment if the production changes.
ALIGNMENT_OK = {name: True for name in (
    [name for group in MC_PROCESSES.values() for name in group['datasets']]
    + [name for names in DATA_PROCESSES.values() for name in names]
)}

FEATURES = [
    'mli_ht', 'mli_lt', 'mli_n_jet', 'mli_n_btag', 'mli_b_score_sum',
    'mli_dr_bb', 'mli_dphi_bb', 'mli_mbb', 'mli_bb_pt', 'mli_mindr_lb',
    'mli_mll', 'mli_dr_ll', 'mli_dphi_ll', 'mli_ll_pt', 'mli_min_dr_llbb',
    'mli_dphi_bb_nu', 'mli_dphi_bb_llMET', 'mli_mllMET', 'mli_mbbllMET',
    'mli_dr_bb_llMET', 'mli_met_pt',
] + [f'mli_{obj}_{var}' for obj in ['lep', 'lep2'] for var in ['pt', 'eta']] \
  + [f'mli_{obj}_{var}' for obj in ['b1', 'b2', 'j1'] for var in ['pt', 'eta', 'b_score']] \
  + [f'mli_{obj}_{var}' for obj in ['fj'] for var in ['pt', 'eta', 'phi', 'mass', 'msoftdrop']]

# Add diagnostic variables here that should be saved in the test fold but NOT used by the NN.
VALIDATION_VARS = []

# Optional physics/analysis selections, applied branch-by-branch during loading to BOTH Data and MC.
# Format: feature -> (lower_bound, upper_bound). Bounds set to None are disabled.
# The lower bound is inclusive; the upper bound is exclusive. Non-finite values are rejected.
#
# IMPORTANT: only put a cut here if it defines the phase space you actually want to validate.
# Do not cut a tail merely to make the scaler look nicer; long-tailed NN inputs use RobustScaler.
SELECTIONS = {
    # 'mli_met_pt': (None, 200.0),
}

# Selection variables must also be read from parquet even when they are not NN inputs.
LOAD_FEATURES = list(dict.fromkeys(FEATURES + VALIDATION_VARS + list(SELECTIONS)))

CHANNELS = ['2mu', '2e']
STAGES = ['before', 'after']
WEIGHT_COLUMNS = {'before': 'weight_uncorrected', 'after': 'weight'}

# NN input features preparation

MAX_EVENTS_PER_CLASS = None
TEST_SIZE = 0.30
VAL_SIZE_WITHIN_TRAINVAL = 0.15
RANDOM_STATE = 0

# NN hyperparameters

HIDDEN = (128, 128, 128)
EPOCHS = 50
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.2
