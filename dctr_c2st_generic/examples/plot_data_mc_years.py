"""Copy to repository root as config_plot_data_mc_years.py and edit marked fields."""
from copy import deepcopy
import runpy
import numpy as np
import c2st_config as old

# Your EXISTING training config; reuse Data lists, exact source paths and selections.
training = runpy.run_path('config_data25_over_data24_2mu.py')['CONFIG']

# Verify these lists separately for each year. Add ALL backgrounds used in analysis.
# The shipped old.MC_PROCESSES contains DY, ttbar DL and ttbb only.
MC24 = deepcopy(old.MC_PROCESSES)
MC25 = deepcopy(old.MC_PROCESSES)

# Same ordered nominal product used to construct weight in dyvr_lib.py.
NOMINAL_FIELDS = [
    'stitched_normalization_weight', 'trigger_weight', 'normalized_pu_weight',
    'muon_id_weight', 'muon_iso_weight', 'electron_weight', 'electron_reco_weight',
    'normalized_ht_njet_nhf_btag_weight', 'normalized_murmuf_envelope_weight',
    'normalized_mur_weight', 'normalized_muf_weight', 'normalized_pdf_weight',
    'normalized_isr_weight', 'normalized_fsr_weight', 'top_pt_theory_weight',
]
# Historical loader treats absent nominal fields as 1 and reports them. Review
# warnings; change optional to False for fields required in your production.
MC_WEIGHT = {'mode': 'product', 'fields': [
    {'producer': 'event_weights', 'column': field, 'optional': True}
    for field in NOMINAL_FIELDS]}

YEARS = {}
for year, side, mc in [('2024', 'base', MC24), ('2025', 'target', MC25)]:
    source = deepcopy(training['sources'][training[f'{side}_source']])
    # REQUIRED: choose the exact official DY producer for EACH year.
    source.setdefault('producers', {})['dy_correction_weight'] = f'/REPLACE/{year}/cf.ProduceColumns/prod__dy_correction_weight_EXACT_VERSION'
    for entry in mc.values():
        for dataset in entry['datasets']:
            source['alignment_ok'][dataset] = False  # Set True only after verifying alignment.
    YEARS[year] = {
        'source': source,
        'data_processes': deepcopy(training[f'{side}_processes']),
        'mc_processes': mc,
        'data_weights': {'mode': 'unit'},  # Repository analysis weight for Data is 1.
        'mc_weights': deepcopy(MC_WEIGHT),
        # Required for DY: never silently omit the official correction.
        'dy_weight': {'producer': 'dy_correction_weight', 'column': 'dy_correction_weight', 'optional': False},
        'luminosity_fb': None,  # EDIT: displayed luminosity only; does not rescale weights.
    }

PLOT_CONFIG = {
    'years': YEARS,
    'bundles': {
        '2mu': 'c2st_artifacts/REPLACE_RUN/2mu/inference',
        '2e': 'c2st_artifacts/REPLACE_RUN/2e/inference',
    },
    'channels': deepcopy(training['channels']),
    'channel_ids': deepcopy(training['channel_ids']),
    'regions': ['dycr'],  # Each region is plotted separately.
    'region_ids': deepcopy(training['region_ids']),
    'selections': deepcopy(training.get('selections', {})),
    'channel_labels': {'2mu': '2 Muon', '2e': '2 Electron'},
    'region_labels': {'dycr': 'DY validation region'},
    'cms_label': 'Work in progress',
    'formats': ['png'],  # Add 'pdf' if desired; three plots per format.
    'inference_chunk_size': 65536,
    'observables': {
        'mli_ll_pt': {'edges': np.linspace(0, 200, 41),
                      'label': r'$p_{T}^{\ell\ell}$ [GeV]', 'log': True, 'flow': 'drop'},
        'mli_met_pt': {'edges': np.linspace(0, 200, 41),
                       'label': r'$p_{T}^{\mathrm{miss}}$ [GeV]', 'log': True, 'flow': 'drop'},
        'mli_n_jet': {'edges': np.arange(-.5, 12.5, 1),
                      'label': 'Jet multiplicity', 'density_per_unit': False, 'log': True},
    },
    'colors': {'DY (ee)': '#ffad05', 'DY (mumu)': '#e8a01a', 'DY (tautau)': '#a86f5b',
               'ttbar DL': '#bf2100', 'ttbb': '#ed6900'},
}
