"""Copy to repository root as config_plot_all_features.py. Run from repository root."""
import runpy
from dctr_c2st_generic.extended_plot_config import extend_plots

# EDIT this filename to your existing, working plot config. Do NOT point to this file.
previous = runpy.run_path('config_plot_data_mc_years.py')['PLOT_CONFIG']
PLOT_CONFIG = extend_plots(previous)
PLOT_CONFIG["regions"] = ["dycr", "ttcr", "ar"]

