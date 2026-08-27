"""
dyvr_lib -- reusable data-loading, region/channel classification, and Data/MC plotting
utilities for the DY-VR C2ST bachelor project (HH->bbWW DL, Run 3).

--------------------------------------------------------------------------------------------
WHY THIS MODULE EXISTS / WHERE IT LIVES (decision, not yet logged as a formal DEC-0xx entry)
--------------------------------------------------------------------------------------------
This was originally inline code in `00_explore_dyvr.ipynb`. Per the 2026-08-06 handoff (§5,
item 1) two placements were on the table:

  (A) framework-free, analysis-adjacent (THIS CHOICE): a plain module next to the notebooks,
      importing only awkward/numpy/pandas/pyarrow/matplotlib -- no `columnflow`/`law` import,
      no dependency on a live `config_inst`.
  (B) inside `hbw` proper (e.g. `hbw/scripts/` or a new `hbw/studies/` subpackage):
      framework-coupled, could use `config_inst` for category-id decoding instead of the
      hardcoded MAIN_IDS/LEP_IDS maps below, and would be installed/versioned with the rest
      of the analysis code.

Chose (A) for two reasons, both stated already in project governance, not newly invented here:
  1. The project plan's explicit design goal is "no columnflow/law exposure for the student"
     -- a module that imports `columnflow`/`law` would defeat that even if the student never
     directly touches those imports, since environment setup (installing `hbw` and its stack)
     becomes a prerequisite just to run the loader.
  2. This module reads *already-produced* task outputs from disk (parquet files under
     `cf_store`/the merged-reduction store); it does not run, require, or depend on a
     `law`/`columnflow` task graph or a live `config_inst`. Putting framework-free code inside
     a framework-coupled package would be a slightly awkward fit either way.

Trade-off accepted knowingly: MAIN_IDS/LEP_IDS (see `classify.py`-equivalent section below)
are a hardcoded, hand-verified snapshot of `hbw/config/categories.py`'s id maps (verified
empirically against real files in the 2026-08-06 session, Section 2 of the notebook) rather
than read live from `config_inst`. If `hbw/config/categories.py`'s id assignment ever changes,
this module's MAIN_IDS/LEP_IDS must be updated by hand and will silently go stale otherwise --
this is the actual cost of choice (A) vs (B). Flagging this explicitly rather than treating it
as solved; worth revisiting as a DEC-0xx entry if/when this module is judged stable enough to
promote into `hbw` itself.

--------------------------------------------------------------------------------------------
DESIGN CHANGE vs. the notebook version: no implicit global state
--------------------------------------------------------------------------------------------
The notebook version of this code relied on three notebook-global variables that every loading
function silently reached into: `reduction_dir`, `producer_dirs` (a flat list, re-searched by
name prefix on every call), and `alignment_ok` (a dict built by running Section 1b once, then
read by `load_process_table` later). That is fine in a single notebook's linear execution
order, but it is a real pitfall for a *library*: a function silently depending on a global that
must have been set by some earlier, unenforced call is exactly the kind of bug that shows up
only when someone reorders cells, imports this in a second notebook, or writes a test.

This module removes that implicit coupling:
  - `discover_store_layout()` returns an explicit `StoreLayout` object (paths resolved once);
    every loading function takes a `layout: StoreLayout` argument instead of reading globals.
  - `check_alignment()` returns the `alignment_ok` dict explicitly; `load_process_table()`
    takes it as an explicit argument instead of reading a notebook global. This makes the
    dependency ("you must run the alignment check before loading, and here is its result")
    visible in the function signature rather than implicit in cell execution order.

--------------------------------------------------------------------------------------------
Sources for the physics/framework facts encoded below (see also the 2026-08-06 handoff, §6)
--------------------------------------------------------------------------------------------
- `columnflow/tasks/reduction.py:101-102,529-532`, `columnflow/tasks/production.py:69-74,110-128`
  -- output file naming (`events_{N}.parquet` / `columns_{N}.parquet`) and the
  `ProvideReducedEvents` requirement that guarantees row-for-row branch alignment.
- `hbw/config/datasets.py:37-51` -- Data is one dataset per era, never a single merged dataset.
- `hbw/config/datasets.py:114-121` -- `stitched_normalization_weight` makes simple
  concatenation across DY mass/jet bins the statistically correct combination.
- `hbw/weight/default.py:323-328` (`default_weight_columns`) -- the full MC event-weight
  product used in the paper's own histograms.
- `hbw/config/categories.py:105-123` (main ids), `:139-186` (lep ids) -- MAIN_IDS/LEP_IDS below.
- `hbw/production/dy_correction_weight.py` -- DY correction is looked up against
  **generator-level** `gen_dilepton_pt`, not the reconstructed `mli_ll_pt`.
- CMS Collaboration, arXiv:2604.02127, Table 1 -- region mll windows (DY VR: 70-110 GeV).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================================
# Section A -- store layout discovery (replaces the notebook's `reduction_dir`/`producer_dirs`
# globals with an explicit, immutable object)
# =============================================================================================

# Canonical short names -> the `prod__*` folder-name prefix that identifies them. Extend this
# dict if a new producer is added to the pipeline; anything not listed here is still reachable
# via `StoreLayout.all_producer_dirs` (e.g. for the Section-1-style inventory scan), just not
# by a friendly `.producer_dirs["..."]` lookup.
_PRODUCER_NAME_PREFIXES = {
    "dl_ml_inputs": "prod__dl_ml_inputs",
    "event_weights": "prod__event_weights",
    "dy_correction_weight": "prod__dy_correction_weight",
    "pre_ml_cats": "prod__pre_ml_cats",
    "cats_ml_multiclass": "prod__cats_ml_multiclass",
}


@dataclass(frozen=True)
class StoreLayout:
    """Resolved, immutable set of paths for one `cf_store` output tree + its matching
    reduction-stage store. Build once with `discover_store_layout()`, then pass this object
    into every loading function below -- no hidden globals.
    """

    store_root: Path
    reduction_dir: Path
    produce_columns_dir: Path
    producer_dirs: dict[str, Path]          # canonical short name -> resolved Path
    all_producer_dirs: list[Path] = field(default_factory=list)  # every prod__* folder found

    def require_producer(self, short_name: str) -> Path:
        """Look up a producer dir by canonical short name, raising a clear error (not a
        confusing KeyError/AttributeError two calls later) if it wasn't found on disk."""
        if short_name not in self.producer_dirs:
            raise KeyError(
                f"Producer '{short_name}' not found under {self.produce_columns_dir}. "
                f"Available: {sorted(self.producer_dirs)}. If this producer was renamed/added, "
                f"update _PRODUCER_NAME_PREFIXES in dyvr_lib.py."
            )
        return self.producer_dirs[short_name]


def discover_store_layout(store_root: Path, reduction_dir: Path) -> StoreLayout:
    """Resolve a StoreLayout from the two root paths the notebook currently hardcodes as
    `STORE_ROOT` / `reduction_dir`. Does not read any data, only lists directories.

    Parameters
    ----------
    store_root : the `cf.ProduceColumns`-containing task-output tree (what the notebook calls
        `STORE_ROOT`).
    reduction_dir : the `cf.MergeReducedEvents` output directory (separate store in this setup
        -- confirmed in the 2026-08-06 session that `cf.ReduceEvents` itself is never used, see
        module docstring).
    """
    produce_columns_dir = store_root / "cf.ProduceColumns"
    all_producer_dirs = (
        sorted(p for p in produce_columns_dir.glob("prod__*") if p.is_dir())
        if produce_columns_dir.exists() else []
    )
    producer_dirs = {}
    for short_name, prefix in _PRODUCER_NAME_PREFIXES.items():
        match = next((d for d in all_producer_dirs if d.name.startswith(prefix)), None)
        if match is not None:
            producer_dirs[short_name] = match
    return StoreLayout(
        store_root=store_root,
        reduction_dir=reduction_dir,
        produce_columns_dir=produce_columns_dir,
        producer_dirs=producer_dirs,
        all_producer_dirs=all_producer_dirs,
    )


# =============================================================================================
# Section B -- low-level file discovery (unchanged logic from the notebook, just parameterized
# on an explicit `base_dir` as before -- these never needed the global-state fix)
# =============================================================================================

def find_one_file(base_dir: Path, process: str, shift: str = "nominal") -> Path | None:
    """One parquet file for (base_dir, process, shift), or None. For schema/sample peeks only
    -- do not use this for anything that needs *all* events of a process."""
    candidates = sorted(base_dir.glob(f"{shift}/{process}*/*.parquet"))
    if not candidates:
        candidates = sorted(base_dir.glob(f"**/{process}*/*.parquet"))
    return candidates[0] if candidates else None


def list_branch_files(base_dir: Path, process: str, shift: str = "nominal") -> dict[int, Path]:
    """{branch_number: file_path}, parsed from `events_{N}.parquet` / `columns_{N}.parquet`."""
    out: dict[int, Path] = {}
    for pattern in (f"{shift}/{process}*/events_*.parquet", f"{shift}/{process}*/columns_*.parquet"):
        for f in base_dir.glob(pattern):
            try:
                branch = int(f.stem.split("_")[-1])
            except ValueError:
                continue
            out[branch] = f
    return out


def branch_row_counts(base_dir: Path, process: str, shift: str = "nominal") -> dict[int, int]:
    """{branch_number: row_count}, read from parquet metadata only (no full read)."""
    return {
        branch: pq.ParquetFile(f).metadata.num_rows
        for branch, f in list_branch_files(base_dir, process, shift).items()
    }


# =============================================================================================
# Section C -- Section 1 / 1b equivalents: column inventory + branch alignment check
# =============================================================================================

def inventory_row(base_dir: Path, label: str, process: str, key_columns: list[str],
                   shift: str = "nominal") -> dict:
    """One row of the Section-1 inventory table: which of `key_columns` are present in one
    sample file for (base_dir, process)."""
    f = find_one_file(base_dir, process, shift)
    row: dict = {"source": label, "process": process, "file": str(f) if f else None}
    if f is None:
        row["status"] = "NO FILE FOUND"
        for c in key_columns:
            row[c] = None
        return row
    top_level = set(pq.read_schema(f).names)
    row["status"] = "ok"
    for c in key_columns:
        row[c] = (c.split(".")[0] in top_level)
    return row


def build_inventory(layout: StoreLayout, processes: list[str], key_columns: list[str],
                     shift: str = "nominal") -> pd.DataFrame:
    """Full Section-1 style inventory DataFrame across `cf.MergeReducedEvents` and every
    discovered producer folder, for every process in `processes`."""
    rows = []
    for proc in processes:
        if layout.reduction_dir.exists():
            rows.append(inventory_row(layout.reduction_dir, "cf.MergeReducedEvents", proc, key_columns, shift))
        for d in layout.all_producer_dirs:
            rows.append(inventory_row(d, d.name, proc, key_columns, shift))
    return pd.DataFrame(rows)


def check_alignment(layout: StoreLayout, processes: list[str], shift: str = "nominal",
                     verbose: bool = True) -> dict[str, bool]:
    """Section-1b equivalent: for each process, confirms `cf.MergeReducedEvents`'s
    `events_{N}.parquet` files correspond row-for-row to every producer's `columns_{N}.parquet`
    (same branch numbers, same row counts per branch). Producer folders are the reference (they
    are what `cf.ProduceColumns` actually consumed); `cf.MergeReducedEvents` is judged against
    them, not the other way around.

    Returns an explicit `{process: bool}` dict -- pass this into `load_process_table()`. Not a
    global; callers must thread it through themselves (see module docstring, Section "no
    implicit global state").
    """
    alignment_ok: dict[str, bool] = {}
    for proc in processes:
        if verbose:
            print(f"=== {proc} ===")

        producer_counts = {d.name: branch_row_counts(d, proc, shift) for d in layout.all_producer_dirs}
        producer_counts = {k: v for k, v in producer_counts.items() if v}
        if not producer_counts:
            if verbose:
                print("  no producer output found for this process, skipping")
            alignment_ok[proc] = False
            if verbose:
                print()
            continue

        producer_branch_sets = {label: set(c) for label, c in producer_counts.items()}
        reference_label, reference_branches = next(iter(producer_branch_sets.items()))
        if not all(b == reference_branches for b in producer_branch_sets.values()):
            if verbose:
                print(f"  WARNING: producer folders disagree on branch numbering: {producer_branch_sets}")
            alignment_ok[proc] = False
            if verbose:
                print()
            continue

        reduction_counts = branch_row_counts(layout.reduction_dir, proc, shift) if layout.reduction_dir.exists() else {}
        if set(reduction_counts) != reference_branches:
            ratio = (len(reduction_counts) / len(reference_branches)) if reference_branches else float("nan")
            if verbose:
                print(f"  MISMATCH: cf.MergeReducedEvents has {len(reduction_counts)} branches, "
                      f"producers have {len(reference_branches)} (ratio {ratio:.2f}). "
                      f"Not safe to merge for this process.")
            alignment_ok[proc] = False
            if verbose:
                print()
            continue

        rows_agree = all(
            reduction_counts.get(b) == prod_counts.get(b)
            for prod_counts in producer_counts.values()
            for b in reference_branches
        )
        alignment_ok[proc] = rows_agree
        if verbose:
            if rows_agree:
                print(f"  OK: {len(reference_branches)} branches, row counts consistent across "
                      f"cf.MergeReducedEvents and all {len(producer_counts)} producer folder(s).")
            else:
                print("  Branch numbers match but row counts differ per branch -- do not use "
                      "without investigating (possible re-run against a different "
                      "reducer/selector version).")
            print()

    if verbose:
        print("alignment_ok:", alignment_ok)
    return alignment_ok


# =============================================================================================
# Section D -- Section 3 equivalent: region / channel classification
# =============================================================================================

# Block-id map for the FLAT, pre-combination `category_ids` written at `cf.MergeReducedEvents`
# (i.e. before `add_categories_production`'s `create_category_combinations` runs). Verified
# empirically against real files, 2026-08-06 session -- see module docstring for the caveat
# about this being a hand-maintained snapshot rather than a live `config_inst` read.
# Source: hbw/config/categories.py:105-123 (main), :139-186 (lep).
MAIN_IDS = {1: "ar", 3: "dycr", 4: "ttcr"}   # "ar" = analysis region (catid_mll_low, "sr" in DL config)
LEP_IDS = {10: "1e", 20: "1mu", 30: "2e", 40: "2mu", 50: "emu", 60: "ge3lep"}


def classify_region_channel_from_category_ids(cat_ids_flat: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """(region, channel) arrays from the flat, pre-combination `category_ids` at the
    `cf.MergeReducedEvents` stage. Vectorized: loops only over the small MAIN_IDS/LEP_IDS maps
    (<=6 iterations), not over events -- branches in this analysis run to ~1-2M events each, so
    a per-event Python loop would be a real bottleneck.
    """
    n = len(cat_ids_flat)
    region = np.full(n, None, dtype=object)
    channel = np.full(n, None, dtype=object)
    for cid, name in MAIN_IDS.items():
        region[ak.to_numpy(ak.any(cat_ids_flat == cid, axis=1))] = name
    for cid, name in LEP_IDS.items():
        channel[ak.to_numpy(ak.any(cat_ids_flat == cid, axis=1))] = name
    return region, channel


def classify_region_channel_from_mll_leptons(
    mll: np.ndarray, n_electron: np.ndarray, n_muon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(region, channel) arrays from raw `mll` + per-event lepton-flavor counts, independent of
    `category_ids` entirely -- the redundant cross-check used to validate the id-based
    classification above (paper Table 1 mll windows; `hbw/selection/dl_remastered.py`
    channel-mask logic, reimplemented on post-baseline-selection lepton counts, so no pT
    threshold needs reimplementing here since baseline selection has already been applied).
    """
    region = np.full(mll.shape, "ttcr", dtype=object)
    region[mll < 110] = "dycr"
    region[mll < 70] = "ar"
    region[mll < 20] = None  # below baseline mll>=20 selection, shouldn't occur post-selection

    channel = np.full(mll.shape, "ge3lep", dtype=object)  # catch-all for unexpected combinations
    channel[(n_electron == 1) & (n_muon == 1)] = "emu"
    channel[(n_muon == 2) & (n_electron == 0)] = "2mu"
    channel[(n_electron == 2) & (n_muon == 0)] = "2e"
    return region, channel


# =============================================================================================
# Section E -- Section 4 equivalent: per-event table loading
# =============================================================================================

# columns pulled from cf.MergeReducedEvents (base, per-event / per-object). gen_dilepton_pt is
# MC-only (hbw/config/config_run2.py:1062, same keep-list as mll) -- absent for Data, handled
# gracefully via _load_fields' missing-field tracking, not assumed present.
REDUCTION_FIELDS = ["mll", "mc_weight", "category_ids", "Muon", "Electron", "gen_dilepton_pt"]

# columns pulled from the prod__dl_ml_inputs producer folder
# DL_ML_INPUT_FIELDS = ["mli_ll_pt", "mli_n_jet", "mli_mbb", "mli_lep_pt", "mli_met_pt", "mli_n_btag"]
DL_ML_INPUT_FIELDS = [
    # event features
    "mli_ht", "mli_lt", "mli_n_jet", "mli_n_btag",
    "mli_b_score_sum",
    # bb system
    "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_bb_pt",
    "mli_mindr_lb",
    # ll system
    "mli_mll", "mli_dr_ll", "mli_dphi_ll", "mli_ll_pt",
    "mli_min_dr_llbb",
    "mli_dphi_bb_nu", "mli_dphi_bb_llMET", "mli_mllMET",
    "mli_mbbllMET", "mli_dr_bb_llMET",
    # VBF features
    # "mli_vbf_deta", "mli_vbf_invmass", "mli_vbf_tag",
    # low-level features
    "mli_met_pt",
] + [
    f"mli_{obj}_{var}"
    for obj in ["b1", "b2", "j1"]
    for var in ["pt", "eta", "b_score"]
] + [
    f"mli_{obj}_{var}"
    for obj in ["lep", "lep2"]
    for var in ["pt", "eta"]
] + [
    f"mli_{obj}_{var}"
    for obj in ["fj"]
    for var in ["pt", "eta", "phi", "mass", "msoftdrop"]
]
# Full MC weight product, hbw/weight/default.py:323-328 (default_weight_columns), minus
# stitched_normalization_weight (loaded first) and dy_correction_weight (applied separately,
# DY-only). Columns absent for a given process (e.g. top_pt_theory_weight for non-ttbar) are
# skipped in the multiplication -- reported, not silently ignored (see load_process_table).
EVENT_WEIGHT_FIELDS = [
    "stitched_normalization_weight", "trigger_weight", "normalized_pu_weight",
    "muon_id_weight", "muon_iso_weight", "electron_weight", "electron_reco_weight",
    "normalized_ht_njet_nhf_btag_weight",
    "normalized_murmuf_envelope_weight", "normalized_mur_weight", "normalized_muf_weight",
    "normalized_pdf_weight", "normalized_isr_weight", "normalized_fsr_weight",
    "top_pt_theory_weight",
]
DY_CORRECTION_FIELDS = ["dy_correction_weight"]
 
 
def _load_fields(base_dir: Path, process: str, branch: int, fields: list[str],
                  shift: str = "nominal") -> tuple[ak.Array | None, list[str]]:
    """Load only `fields` that actually exist in the schema, from one branch's parquet file for
    (base_dir, process). Returns (array, missing_fields) so callers know what was skipped."""
    branch_files = list_branch_files(base_dir, process, shift)
    if branch not in branch_files:
        return None, fields
    schema_names = set(pq.read_schema(branch_files[branch]).names)
    present = [f for f in fields if f in schema_names]
    missing = [f for f in fields if f not in schema_names]
    table = pq.read_table(branch_files[branch], columns=present)
    return ak.from_arrow(table), missing
 
 
def load_process_table(
    layout: StoreLayout,
    processes: str | list[str],
    alignment_ok: dict[str, bool],
    shift: str = "nominal",
    is_dy: bool = False,
    is_data: bool = False,
    verbose: bool = True,
    exclude_from_uncorrected: tuple[str, ...] = (),
    feature_fields: list[str] | None = None,
    validate_classification: bool = False,
    keep_region: str | None = None,
    keep_channels: tuple[str, ...] | None = None,
    selections: dict[str, tuple[float | None, float | None]] | None = None,
    compact_dtypes: bool = True,
) -> pd.DataFrame:
    """Load one logical process into a compact per-event DataFrame.

    The production path reads only requested ML features, filters each parquet
    branch before retaining it, and never constructs one giant Awkward array.
    """
    if isinstance(processes, str):
        processes = [processes]
    if feature_fields is None:
        feature_fields = DL_ML_INPUT_FIELDS
    selections = selections or {}

    # Selection variables need to be physically loaded even when they are not NN inputs.
    feature_fields = list(dict.fromkeys([*feature_fields, *selections.keys()]))

    unknown = set(feature_fields) - set(DL_ML_INPUT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown DL-ML feature(s): {sorted(unknown)}")

    dl_ml_inputs_dir = layout.require_producer("dl_ml_inputs")
    event_weights_dir = layout.require_producer("event_weights")
    dy_corr_dir = layout.producer_dirs.get("dy_correction_weight")

    reduction_fields = ["category_ids"]
    if validate_classification:
        reduction_fields += ["mll", "Muon", "Electron", "gen_dilepton_pt"]

    per_branch_frames: list[pd.DataFrame] = []
    missing_weight_cols_seen: set[str] = set()
    n_seen = 0
    n_kept = 0

    region_id = None
    if keep_region is not None:
        region_id = next((cid for cid, name in MAIN_IDS.items() if name == keep_region), None)
        if region_id is None:
            raise ValueError(f"Unknown region {keep_region!r}")

    requested_channels = tuple(keep_channels) if keep_channels is not None else None
    channel_ids = {}
    if requested_channels is not None:
        for channel_name in requested_channels:
            cid = next((cid for cid, name in LEP_IDS.items() if name == channel_name), None)
            if cid is None:
                raise ValueError(f"Unknown channel {channel_name!r}")
            channel_ids[channel_name] = cid

    for process in processes:
        if not alignment_ok.get(process, False):
            if verbose:
                print(f"  WARNING: '{process}' did not pass the alignment check -- SKIPPING it entirely.")
            continue

        reduction_files = list_branch_files(layout.reduction_dir, process, shift)
        dy_files = list_branch_files(dy_corr_dir, process, shift) if (is_dy and dy_corr_dir) else {}

        for branch in sorted(reduction_files):
            base, _ = _load_fields(layout.reduction_dir, process, branch, reduction_fields, shift)
            if base is None:
                continue

            n_branch = len(base)
            n_seen += n_branch
            cat_ids = base["category_ids"]
            mask = np.ones(n_branch, dtype=bool)

            if region_id is not None:
                mask &= ak.to_numpy(ak.any(cat_ids == region_id, axis=1))

            branch_channel_masks = {}
            if requested_channels is not None:
                channel_union = np.zeros(n_branch, dtype=bool)
                for channel_name, cid in channel_ids.items():
                    cmask = ak.to_numpy(ak.any(cat_ids == cid, axis=1))
                    branch_channel_masks[channel_name] = cmask
                    channel_union |= cmask
                mask &= channel_union

            if not mask.any():
                continue

            mli, mli_missing = _load_fields(dl_ml_inputs_dir, process, branch, feature_fields, shift)
            if mli is None or mli_missing:
                raise KeyError(f"Missing requested ML fields for {process}, branch {branch}: {mli_missing}")

            # Apply configured physics selections before retaining any feature/weight arrays.
            # This is deliberately loader-side so rejected events never enter the large pandas tables.
            for field, bounds in selections.items():
                if len(bounds) != 2:
                    raise ValueError(
                        f"Selection for {field!r} must be a (lower, upper) pair, got {bounds!r}"
                    )
                lower, upper = bounds
                values = ak.to_numpy(mli[field])
                field_mask = np.isfinite(values)
                if lower is not None:
                    field_mask &= values >= lower
                if upper is not None:
                    field_mask &= values < upper
                mask &= field_mask

            nkeep = int(mask.sum())
            if nkeep == 0:
                continue

            out_dict = {}
            if keep_region is None:
                region, _ = classify_region_channel_from_category_ids(cat_ids)
                out_dict["region"] = pd.Categorical(region[mask])
            else:
                out_dict["region"] = pd.Categorical(
                    np.repeat(keep_region, nkeep), categories=[keep_region]
                )

            if requested_channels is None:
                _, channel = classify_region_channel_from_category_ids(cat_ids)
                out_dict["channel"] = pd.Categorical(channel[mask])
            else:
                retained = np.flatnonzero(mask)
                codes = np.full(nkeep, -1, dtype=np.int8)
                for code, channel_name in enumerate(requested_channels):
                    codes[branch_channel_masks[channel_name][retained]] = code
                out_dict["channel"] = pd.Categorical.from_codes(codes, categories=list(requested_channels))

            for f in feature_fields:
                values = ak.to_numpy(mli[f])[mask]
                out_dict[f] = values.astype(np.float32, copy=False) if compact_dtypes else values

            dtype = np.float32 if compact_dtypes else np.float64
            if is_data:
                out_dict["weight_uncorrected"] = np.ones(nkeep, dtype=dtype)
                out_dict["weight"] = np.ones(nkeep, dtype=dtype)
            else:
                ew, missing = _load_fields(event_weights_dir, process, branch, EVENT_WEIGHT_FIELDS, shift)
                missing_weight_cols_seen |= set(missing)
                weight_full = np.ones(n_branch, dtype=dtype)
                weight_unc = np.ones(n_branch, dtype=dtype)
                for col in EVENT_WEIGHT_FIELDS:
                    if col in missing:
                        continue
                    values = ak.to_numpy(ew[col]).astype(dtype, copy=False)
                    weight_full *= values
                    if col not in exclude_from_uncorrected:
                        weight_unc *= values

                weight_after = weight_full
                if is_dy and dy_corr_dir is not None and branch in dy_files:
                    dy, dy_missing = _load_fields(dy_corr_dir, process, branch, DY_CORRECTION_FIELDS, shift)
                    if "dy_correction_weight" not in dy_missing:
                        corr = ak.to_numpy(dy["dy_correction_weight"]).astype(dtype, copy=False)
                        weight_after = weight_full * corr

                out_dict["weight_uncorrected"] = weight_unc[mask]
                out_dict["weight"] = weight_after[mask]

            if validate_classification:
                region_a, channel_a = classify_region_channel_from_category_ids(cat_ids)
                region_b, channel_b = classify_region_channel_from_mll_leptons(
                    ak.to_numpy(base["mll"]),
                    ak.to_numpy(ak.num(base["Electron"]["pt"], axis=1)),
                    ak.to_numpy(ak.num(base["Muon"]["pt"], axis=1)),
                )
                if verbose:
                    print(
                        f"[{process}, branch {branch}] region agreement: {np.mean(region_a == region_b):.4%}, "
                        f"channel agreement: {np.mean(channel_a == channel_b):.4%}"
                    )

            per_branch_frames.append(pd.DataFrame(out_dict))
            n_kept += nkeep

            del base, cat_ids, mli, out_dict

    if missing_weight_cols_seen and verbose:
        print(
            "  note: weight columns not found for at least one branch/process "
            f"(treated as factor 1): {sorted(missing_weight_cols_seen)}"
        )

    if not per_branch_frames:
        return pd.DataFrame(columns=["region", "channel", *feature_fields, "weight_uncorrected", "weight"])

    out = pd.concat(per_branch_frames, ignore_index=True)
    if verbose:
        print(f"[{processes}] kept {n_kept:_} / {n_seen:_} events after loader-side selection")
    return out


def load_all(
    layout: StoreLayout,
    mc_processes: dict[str, dict],
    data_processes: dict[str, list[str]],
    alignment_ok: dict[str, bool],
    shift: str = "nominal",
    verbose: bool = True,
    exclude_from_uncorrected: tuple[str, ...] = (),
    feature_fields: list[str] | None = None,
    validate_classification: bool = False,
    keep_region: str | None = None,
    keep_channels: tuple[str, ...] | None = None,
    selections: dict[str, tuple[float | None, float | None]] | None = None,
    compact_dtypes: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load all configured groups with optional branch-level selection."""
    tables = {}
    common = dict(
        alignment_ok=alignment_ok,
        shift=shift,
        verbose=verbose,
        exclude_from_uncorrected=exclude_from_uncorrected,
        feature_fields=feature_fields,
        validate_classification=validate_classification,
        keep_region=keep_region,
        keep_channels=keep_channels,
        selections=selections,
        compact_dtypes=compact_dtypes,
    )
    for label, group in mc_processes.items():
        tables[label] = load_process_table(
            layout, group["datasets"], is_dy=group["is_dy"], is_data=False, **common
        )
    for label, era_datasets in data_processes.items():
        tables[label] = load_process_table(
            layout, era_datasets, is_dy=False, is_data=True, **common
        )
    return tables
# Section F -- Data/MC comparison plotting
# =============================================================================================

def weighted_hist(values: np.ndarray, weights: np.ndarray, bins) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram + its statistical uncertainty per bin (sqrt(sum(w_i^2)), the correct weighted
    statistical uncertainty -- NOT sqrt(N), which is only valid for weight==1, i.e. Data).
    Returns (counts, uncertainty, bin_edges).
    """
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    sumw2, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    return counts, np.sqrt(sumw2), edges


def plot_data_mc_ratio(
    mc_dfs, data_dfs, feature: str, bins, weight_col: str = "weight",
    xlabel: str | None = None, title: str | None = None, logy: bool = True,
    density: bool = False, figsize=(7, 6), axes=None,
):
    """
    Generic Data/MC comparison with a ratio panel underneath.

    Parameters
    ----------
    ...
    axes : tuple of (ax_main, ax_ratio), optional
        Existing matplotlib axes to draw on. If None, a new 2-panel figure
        is created.
    """

    import matplotlib.pyplot as plt

    if axes is None:
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
    else:
        ax_main, ax_ratio = axes
        fig = ax_main.figure

    centers = 0.5 * (bins[1:] + bins[:-1])
    widths = np.diff(bins)

    # First pass: raw (un-normalized) counts per MC group, plus the running total.
    mc_counts_per_label = []
    mc_stack_total = np.zeros(len(bins) - 1)
    mc_var_total = np.zeros(len(bins) - 1)

    for df, label in mc_dfs:
        if len(df) == 0:
            continue

        counts, err, _ = weighted_hist(
            df[feature].to_numpy(),
            df[weight_col].to_numpy(),
            bins,
        )

        mc_counts_per_label.append((label, counts))
        mc_stack_total += counts
        mc_var_total += err ** 2

    mc_err_total = np.sqrt(mc_var_total)

    if data_dfs:
        data_vals = np.concatenate(
            [df[feature].to_numpy() for df in data_dfs if len(df)]
        )
        data_counts, _ = np.histogram(data_vals, bins=bins)
    else:
        data_counts = np.zeros(len(bins) - 1)

    data_err = np.sqrt(data_counts)

    mc_norm = (
        mc_stack_total.sum()
        if (density and mc_stack_total.sum())
        else 1.0
    )
    data_norm = (
        data_counts.sum()
        if (density and data_counts.sum())
        else 1.0
    )

    mc_plot = mc_stack_total / mc_norm
    mc_err_plot = mc_err_total / mc_norm
    data_plot = data_counts / data_norm
    data_err_plot = data_err / data_norm

    # Draw MC stack
    bottom = np.zeros(len(bins) - 1)

    for label, counts in mc_counts_per_label:
        ax_main.bar(
            centers,
            counts / mc_norm,
            width=widths,
            bottom=bottom,
            label=label,
            alpha=0.8,
        )
        bottom = bottom + counts / mc_norm

    # Draw Data
    if data_dfs:
        ax_main.errorbar(
            centers,
            data_plot,
            yerr=data_err_plot,
            fmt="ko",
            ms=3,
            label="Data",
        )

    if title:
        ax_main.set_title(title, fontsize=10)

    ax_main.set_ylabel("Normalized" if density else "Events / bin")

    if logy:
        ax_main.set_yscale("log")

    ax_main.legend(fontsize=7, ncol=2)

    # Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            mc_plot > 0,
            data_plot / mc_plot,
            np.nan,
        )

        ratio_err = np.where(
            mc_plot > 0,
            ratio * np.sqrt(
                (data_err_plot / np.where(data_plot > 0, data_plot, np.nan)) ** 2
                + (mc_err_plot / np.where(mc_plot > 0, mc_plot, np.nan)) ** 2
            ),
            np.nan,
        )

    ax_ratio.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        fmt="ko",
        ms=3,
    )

    ax_ratio.axhline(1.0, color="gray", ls="--", lw=1)
    ax_ratio.set_ylim(0.75, 1.25)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel(xlabel if xlabel is not None else feature)

    if axes is None:
        fig.tight_layout()

    return fig, (ax_main, ax_ratio)
