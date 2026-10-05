"""Historical DY vs (Data - non-DY MC), correcting DY only."""
from copy import deepcopy
from dctr_c2st_generic.examples.legacy_inclusive import CONFIG as INCLUSIVE
CONFIG = deepcopy(INCLUSIVE)
CONFIG["description"] = "Compatibility: signed Data-minus-non-DY target vs DY"
CONFIG["subtract_processes"] = [name for name, entry in CONFIG["base_processes"].items() if not entry.get("is_dy", False)]
