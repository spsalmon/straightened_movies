# CLAUDE.md — towbintools_pipeline

A modular image-analysis pipeline for *C. elegans* time-lapse microscopy. Built on the lab's **towbintools** Python library. Runs as SLURM jobs that chain composable "building blocks", with a Shiny GUI for manual annotation.

---

## Environment — CRITICAL

**Always run inside the `towbintools` micromamba environment.** It already exists on this system — do not reinstall, recreate, or modify it without asking.

```bash
~/.local/bin/micromamba run -n towbintools <command>
```

The pipeline itself hardcodes this path in many places (`run -n towbintools` is embedded in generated SLURM scripts, see `pipeline_scripts/utils.py:393` `run_command`).

---

## Quick navigation — "where do I look for X?"

| If the task is about… | Open these files |
|---|---|
| Pipeline entry point / config loading | `pipeline_scripts/init_pipeline.py` |
| How blocks are chained between SLURM jobs | `pipeline_scripts/block_linker.py` |
| Adding a new block, validating config | `pipeline_scripts/building_blocks.py` (`OPTIONS_MAP`, `DEFAULT_OPTIONS`, `parse_and_create_building_blocks`) |
| Output naming convention (`chN_seg`, `_str`, etc.) | `pipeline_scripts/utils.py:164` `get_output_name` |
| SLURM script generation | `pipeline_scripts/utils.py:428` `create_sbatch_file` |
| Filemap I/O | `towbintools.foundation.file_handling` (external lib) — `read_filemap`, `write_filemap`, `get_dir_filemap`, `add_dir_to_experiment_filemap` |
| GUI app entry | `gui/app.py` → `gui/run.py` (CLI) |
| GUI reactive logic / save-on-exit | `gui/app_components/server.py` |
| GUI column inference, QC placeholder, "values at molt" | `gui/app_components/backend.py` |
| GUI layout (panels, buttons) | `gui/app_components/ui.py`, `gui/app_components/ui_components.py` |
| Training segmentation models | `training/segmentation/` |
| Training QC classifiers | `training/classification/` |
| ND2 → TIFF, SQUID/MATLAB conversion | `util_scripts/convert_*.py` |
| Generate filemap from a directory | `util_scripts/generate_experiment_filemap.py` |
| Per-block parameter docs | `book/building_blocks/*.ipynb` |
| Custom user scripts (e.g. pumping rate) | `custom_scripts/` |

---

## Running the pipeline

### Standard (SLURM)

```bash
bash run_pipeline.sh -c configs/my_config.yaml
```

`run_pipeline.sh` checks for git updates, then submits `_sbatch_pipeline.sh` via `sbatch`. The job:
1. Copies the config to `temp_files/pipeline_<job_id>/`
2. Saves git info + Python/towbintools versions to `git_info.txt`
3. Redirects SLURM logs into the temp dir
4. Runs `python3 -m pipeline_scripts.init_pipeline -c <config> --temp_dir <temp>`

### Direct (development)

```bash
~/.local/bin/micromamba run -n towbintools python3 -m pipeline_scripts.init_pipeline -c configs/test_config.yaml
```

### How blocks actually run (important for debugging)

Each block submits **its own** SLURM job. Sequential chaining is achieved via a "block linker": each block's sbatch script ends by calling `block_linker.py`, which:
- Reads the **progress tracker pickle** in `temp_files/<run>/pickles/progress_tracker.pkl`
- Updates the experiment filemap with the previous block's output (adds new columns)
- Submits the next block

This means a "pipeline run" is N+1 SLURM jobs, not one. To debug a stuck run: check `temp_files/<run>/sbatch_output/`, then `pickles/`, then `batch/` for the generated sbatch scripts.

---

## Building blocks — string names and outputs

Use these **exact strings** in `building_blocks:` (see `OPTIONS_MAP` in `building_blocks.py:14`):

| Name | Script | Output type | Output naming |
|---|---|---|---|
| `segmentation` | `learning_based_segment.py` (DL/cellpose) or `non_learning_segment.py` (threshold/edge) | image dir | `analysis/ch{N+1}_seg/` (channels are 1-indexed in output names, 0-indexed in config) |
| `straightening` | `straighten.py` | image dir | `<source>_str/` (e.g. `analysis/ch2_seg_str`) |
| `morphology_computation` | `compute_morphology.py` | report file | `analysis/report/<masks>_morphology.<csv\|parquet>` |
| `quality_control` | `quality_control.py` | report file | `analysis/report/<masks>_qc.<ext>` |
| `fluorescence_quantification` | `quantify_fluorescence.py` | report file | `analysis/report/ch{N+1}_fluo_quant_on_<masks>.<ext>` |
| `molt_detection` | `detect_molts.py` | report file | `analysis/report/ecdysis.<ext>` |
| `custom` | user-provided | `subdir` or `csv` | controlled by `custom_script_name` |

**Channel indexing gotcha**: `segmentation_channels: [[1], [0]]` means *the first block segments channel index 1 (i.e. the 2nd channel)*, and the output column is named `ch2_seg` (1-indexed in the name). `get_output_name` does `+1` when building the name.

The output path-building logic lives in `pipeline_scripts/utils.py:164`. When extending, follow the same pattern so downstream blocks can reference outputs predictably (e.g. `straightening_masks: 'analysis/ch2_seg'`).

### Per-block options

`OPTIONS_MAP` (`building_blocks.py:14`) lists every accepted YAML key per block. `DEFAULT_OPTIONS` (`building_blocks.py:75`) lists which are optional and their fallback. Keys not in `OPTIONS_MAP` raise `ValueError` for the block. Add new keys there when extending a block.

### Distribution semantics (important)

Each parameter is a list. If `building_blocks` has 2 `segmentation` entries and you set `segmentation_method: ["deep_learning"]` (length 1), both blocks use the same value. If you set `["deep_learning", "edge_based"]` (length matches), they're distributed in order. Length must be 1 or N; otherwise `parse_and_create_building_blocks` asserts.

Use `null` (YAML) for empty values.

### `rerun_<block>` flags

`False` = skip outputs that already exist. For per-image blocks (segmentation, straightening), this is per-file. For report blocks (morphology, qc, etc.), the whole block is skipped if the report file exists.

---

## Filemap — the central data structure

The **filemap** is a Polars DataFrame (not pandas) tracking every input and output. Persisted at `analysis/report/analysis_filemap.{csv,parquet}` (format from `report_format` in config).

### Column conventions

| Column | Source | Meaning |
|---|---|---|
| `Time`, `Point` | parsed from filenames via `time_regex`/`point_regex` | indices |
| `ExperimentTime` | acquisition timestamps (seconds) | filled if `get_experiment_time: True` |
| `raw` (or `raw_dir_name`) | raw image dir | path to raw OME-TIFF |
| `analysis/chN_seg` | segmentation block | path to mask |
| `analysis/chN_seg_str` | straightening block | straightened mask |
| `analysis/chN_raw_str` | straightening of raw | straightened raw image |
| `<feature>` (e.g. `ch2_seg_str_volume`) | morphology / fluorescence | scalar per row |
| `<col>_at_HatchTime`, `_at_M1`…`_at_M4` | GUI / molt detection | feature values at developmental events |
| `<mask>_qc` | QC block | classification label (`worm`, `egg`, `error`, …) |
| `HatchTime`, `M1`–`M4` | molt detection / GUI annotation | time index of event |
| `Death`, `Arrest`, `Ignore` | GUI annotation | manual flags |

When a filemap is loaded, **legacy `worm_type` columns are auto-renamed to `qc`** (`gui/app_components/backend.py:65`).

### "Annotated" filemap

The GUI saves to `analysis_filemap_annotated.<ext>` next to the original. By default the GUI re-opens the annotated copy if it exists (`OPEN_ANNOTATED=1`). The pipeline's `init_pipeline` will open the annotated version only if `overwrite_annotated_filemap: True` is set in config. A backup is written to the same folder on every GUI open via `get_backup_path` (`backend.py:25`).

---

## Project layout

```
pipeline_scripts/        Core pipeline (init, blocks, utils, linker, per-block scripts)
gui/                     Shiny annotation app
  app.py, run.py         Entry points (env-var driven)
  app_components/        backend.py, server.py, ui.py, ui_components.py
  tests/                 pytest tests (test_backend_minimal.py)
configs/                 YAML configs (config.yaml is the reset-on-update template)
training/
  segmentation/          Train/fine-tune DL segmentation models, dataset gathering
  classification/        Train QC XGBoost / DL classifiers
  utils/                 Eval, metrics, accuracy
util_scripts/            ND2/SQUID/MATLAB → TIFF conversion, filemap generation
custom_scripts/          User scripts invoked via "custom" block
models/                  Pre-trained checkpoints (molt detection, 10x_body_qc, 10x_pharynx_qc)
book/                    MyST/Jupyter Book documentation source
  Landing.md, getting_started/, usage/, building_blocks/, training/
analysis_and_plots/      Notebooks for downstream analysis
requirements/            environment.yml, conda-lock files
sbatch_output/           SLURM stdout/stderr (initial location, then moved to temp_files)
temp_files/              Per-run dirs: pickles/, batch/, sbatch_output/, config copy, git_info
```

---

## Tests

```bash
~/.local/bin/micromamba run -n towbintools pytest gui/tests/
```

Coverage is minimal: `gui/tests/test_backend_minimal.py` validates filemap population, QC placeholders, and feature column handling. `conftest.py` puts `gui/` on `sys.path`. There is **no test suite for `pipeline_scripts/`** — verify pipeline changes by running `configs/test_config.yaml` against a small dataset.

Pre-commit hooks (`.pre-commit-config.yaml`): `reorder-python-imports`, `black`, `flake8`, `pyupgrade --py39-plus`, YAML check.

---

## The annotation GUI

Launch:

```bash
bash launch_gui.sh   # edit FILEMAP_PATH inside first
```

Or directly:

```bash
~/.local/bin/micromamba run -n towbintools python gui/run.py --filemap <path> [--no-annotated] [--recompute] [--port N]
```

Env vars read by `app.py`: `FILEMAP_PATH`, `OPEN_ANNOTATED` (default 1), `RECOMPUTE_FEATURES` (default 0).

### Layout (two columns)

**Left — molt annotator (`create_molt_annotator` in `ui.py:12`)**
- Plotly time-series of the chosen feature for the current Point
- Click a marker → jumps to that Time in the image panel
- Buttons per ecdysis: `HatchTime`, `M1`, `M2`, `M3`, `M4` (annotate or peg the value)
- Action buttons: `Arrest`, `Dead`, `Ignore After`, `Ignore Point`
- Custom column annotation (select existing or type a new one)
- Annotation import: `.csv`, `.parquet`, or legacy MATLAB `.mat` (key conversion in `backend.py:17` `KEY_CONVERSION_MAP`)
- Plot height/width sliders, log-scale toggle

**Right — image viewer (`create_timepoint_selector` in `ui.py:83`)**
- Image preview via `towbintools.foundation.image_handling`
- Time / Point navigators (prev/next + selectize)
- Channel + overlay-channel + segmentation-overlay dropdowns
- Feature-to-plot selector (sets the y-axis on the left)
- Save button

### Marker shapes (data semantics)

Triangles = QC-flagged errors. Squares = eggs. Circles = valid worms. (`set_marker_shape` in `backend.py:539`.)

### Save behaviour

`server.py` registers a save hook called from `app.py`'s `atexit`. Manual `Save` button writes the annotated filemap to `<filemap>_annotated.<ext>`. **Backups are written every time a filemap is opened** (`get_backup_path` increments `_v1`, `_v2`, …).

### Common debugging entry points

- "No qc column found" → `populate_column_choices` (`backend.py:108`) inserts a `placeholder_qc` column with all `"worm"`.
- "No feature column found" → same function inserts `placeholder_feature = 1.0`.
- Default plotted column is the first column containing `volume` (without `_at_`); falls back to first feature column.
- Channels inferred from raw OME-TIFF shape in `infer_n_channels` (`backend.py:92`): `ndim==4` → axis 1; `ndim==3` → axis 0; `ndim==2` → 1.

---

## Configuration cheat sheet

Reference template: `configs/config.yaml`. Other configs in `configs/` are real working setups (per researcher / per scope).

```yaml
# Required core
experiment_dir: "/path/to/experiment"
analysis_dir_name: "analysis"
raw_dir_name: "raw"
report_format: "parquet"          # or "csv"
pixelsize: [ 0.65 ]
get_experiment_time: True
time_regex: 'Time(\d+)'
point_regex: 'Point(\d+)'

# SLURM
sbatch_memory: 64G
sbatch_time: 0-48:00:00
sbatch_cpus: 32
sbatch_gpus: "rtx6000:1"          # only allocated to GPU-requiring blocks

# Pipeline
building_blocks:
  - "segmentation"
  - "straightening"
  - "morphology_computation"

# Per-block (see OPTIONS_MAP for the full list)
segmentation_column: [ 'raw' ]
segmentation_method: [ "deep_learning" ]
segmentation_channels: [ [ 1 ] ]
model_path: [ "/path/to/model.ckpt" ]
batch_size: [ 4 ]
```

### Subdirectory mode

If `<experiment_dir>/<raw_dir_name>/` contains subdirectories (not images directly), the pipeline runs once per subdir and mirrors the layout in `analysis/`. This is detected by `get_experiment_subdirs` (`utils.py:87`).

### "no timepoints" fallback

If `get_dir_filemap` returns empty (filenames don't match the regexes), the pipeline falls back to listing files as a single `ImagePath` column and sets `no_timepoints: True`. See `init_pipeline.py:101`.

---

## Gotchas

- **Channels are 0-indexed in config (`segmentation_channels: [[1]]` = 2nd channel) but 1-indexed in output names (`ch2_seg`).** `+1` happens in `get_output_name`.
- **The filemap is Polars, not pandas.** Use `pl.col(...)`, `.with_columns`, `.select`. `to_numpy().squeeze()` is the common bridge.
- **Block scripts run in separate SLURM jobs.** A single block failure orphans the run — the linker won't fire. Check `temp_files/<run>/sbatch_output/` for the failing block's logs.
- **`configs/config.yaml` is reset on every `update_pipeline.sh` run.** Copy it to a different name before editing for real work.
- **`utils.py` is ~3138 lines** — when extending, prefer adding focused helpers rather than mega-functions. Use `grep -n "^def "` to navigate.
- **`towbintools` (the underlying library) is external.** When a function isn't in this repo, look in the `towbintools` package — file-handling lives in `towbintools.foundation.file_handling`, image I/O in `towbintools.foundation.image_handling`.
- **The `classification` block name was renamed to `quality_control`.** A friendly error is raised if the old name is used (`building_blocks.py:734`).
- **`molt_detection_volume` is deprecated** in favour of `molt_detection_columns`.
- **GUI can't open a Parquet without `pyarrow`** — already in the env, but worth knowing if you see import errors.

---

## Documentation

The `book/` directory is the source for the public Jupyter Book at `https://spsalmon.github.io/towbintools_pipeline/`. Per-block parameter references are notebooks in `book/building_blocks/*.ipynb`. When changing block behaviour, update the matching notebook.

---

## Git commit guidelines

- Imperative mood, specific subject under ~72 chars
- Body explains the *why* (constraint, bug, motivation), not the *what*
- Use a type prefix when it adds clarity: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- **Do not mention AI tools, assistants, or "Claude" in commit messages**
- No filler ("minor tweaks", "updates"). One commit per logical change.

Examples:

```
feat: add QC placeholder generation for experiments without QC data

fix: prevent filemap column duplication when rerunning segmentation

refactor: extract time-parsing logic from utils into a dedicated module

test: cover filemap population for minimal (no-feature, no-QC) configs
```
