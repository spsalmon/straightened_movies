# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`towbintools` is a Python library for biological image analysis, developed for the Towbin Lab at the University of Bern. It is primarily used to analyze *C. elegans* (nematode) timelapse microscopy images and integrates tightly with [towbintools_pipeline](https://github.com/spsalmon/towbintools_pipeline). It is published to PyPI.

**Domain vocabulary:**
- **Worm** — *C. elegans* nematode imaged under a microscope
- **Molt** — larval stage transition (L1 → L2 → L3 → L4 → adult); detecting these from volume time series is a core analysis task
- **Straightening** — geometrically unrolling a curved worm into a rectangular image using a fitted midline spline; required before computing length/volume/width features
- **Pixelsize** — physical size of one pixel in µm; passed explicitly to all feature computations
- **Timelapse / time series** — repeated imaging of the same worm over time; data is stored as arrays or DataFrames indexed by timepoint

## Build & Install

```bash
# Install from source (editable dev install)
pip install -e ".[dev]"

# Build distributable
python -m build
pip install dist/*.whl

# Install pre-commit hooks
pre-commit install
```

There are no test files in this repository — new code is validated by running it in context.

## Testing

**IMPORTANT: All tests and code validation MUST be run through the `towbintools` micromamba environment.** Always prefix execution commands with `micromamba run -n towbintools`, for example:

```bash
micromamba run -n towbintools python my_script.py
micromamba run -n towbintools python -c "import towbintools"
```

Do not run tests or validation scripts with a bare `python`/`pip` — they must go through the `towbintools` micromamba environment.

## Linting & Formatting

Pre-commit hooks enforce style on every commit. To run manually:

```bash
pre-commit run --all-files
```

Stack: **black** (formatter, max line 88), **flake8** (E501/E203 ignored), **reorder-python-imports** (alphabetical within groups, py39+), **pyupgrade** (py39+). mypy is configured but disabled in pre-commit.

## Publishing

Releases are published to PyPI automatically via GitHub Actions when a GitHub Release is created. The version is set in `pyproject.toml`.

## Package Architecture

The package is organized into domain-specific submodules under `towbintools/`:

| Submodule | Purpose |
|---|---|
| `foundation` | Core image I/O, binary mask ops, worm feature extraction, z-stack handling, file utilities |
| `segmentation` | Classical (edge/threshold-based) worm segmentation |
| `straightening` | Spline-based image warping to "straighten" curved worms; exposes `Warper` |
| `deep_learning` | PyTorch Lightning models for segmentation, classification, and keypoint detection |
| `classification` | QC and classification tooling |
| `quantification` | Fluorescence quantification from masks/images |
| `data_analysis` | Time-series processing (smoothing, interpolation, growth rate); molt detection |
| `plotting` | seaborn/matplotlib wrappers for biological data visualization |

**Dependency rule:** `foundation` is the base — all other submodules import from it, never the reverse. `straightening` imports from `foundation.binary_image` indirectly; `foundation.worm_features` imports `straightening.Warper`.

### Deep learning models

Models are PyTorch Lightning (`pl.LightningModule`) subclasses, built on `segmentation_models_pytorch` encoders and `timm` backbones. Factory functions in `deep_learning_tools.py` accept an optional `checkpoint_path` to load from `.ckpt` (when provided, all other args are ignored). Training uses F1 score as the primary metric, logged via `self.log()` with `sync_dist=True`.

### Data flow for worm analysis

A typical analysis pipeline looks like:
1. Read TIFF with `foundation.image_handling.read_tiff_file` (OME-TIFF metadata parsed via `ome-types`)
2. Segment with `segmentation.segment_image` (classical) or a `deep_learning` model
3. Straighten with `straightening.Warper.from_img(mask, mask_for_midline)` → `warper.warp_2D_img(...)`
4. Compute features with `foundation.worm_features.compute_mask_morphological_features(straightened_mask, pixelsize, features)`
5. Store results in pandas/polars DataFrames; run time-series analysis in `data_analysis`

## Coding Style

### Type annotations

All public functions are fully annotated. Use py39+ lowercase generics (`list[str]`, `dict[str, float]`, `tuple[bool, int]`). Use `T | None` for optional types in new code (older files use `Optional[T]` from `typing`). Use `np.ndarray` for all image and array parameters.

### Docstrings

Every public function has a docstring. Follow this exact format:

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    One-sentence summary of what the function does.

    Optionally one or two sentences of elaboration on the approach/behavior.

    Parameters:
        param1 (type): Description. Include shape info for arrays, e.g. ``(..., H, W)``.
        param2 (type): Description. (default: value)

    Returns:
        return_type: Description of what is returned.

    Raises:
        ValueError: When and why this is raised.
    """
```

Key details:
- Parameter descriptions use `(type): description` inline, not a separate `type` field
- Default values are always noted as `(default: value)` at the end of the description
- Array shapes are documented as `(..., H, W)` with leading `...` to indicate broadcasting over extra dims
- `Raises:` section only when the function explicitly raises
- Use double backticks for code in docstrings: `` ``"mean"`` ``, `` ``True`` ``

### Imports

`reorder-python-imports` enforces ordering automatically: stdlib → third-party → local, alphabetical within groups. Always use explicit name imports: `from tifffile import imread` not `import tifffile`. Local imports use the full package path: `from towbintools.foundation import image_handling`.

### Error handling and return values

- Raise `ValueError` with a descriptive message for invalid user-facing arguments
- In computational functions that may fail on bad image data, catch exceptions, print a warning, and return `np.nan` rather than propagating the exception
- At I/O boundaries (file reading, OME-TIFF metadata), use `try/except`, print the error with context, and return `None`
- Use `assert` only for internal invariants (length checks between paired arrays), not for user input validation

### Image array conventions

- Shape is always `(..., H, W)` — spatial dimensions last, any leading dims are channels or z/t
- Binary masks are `np.uint8` with foreground=1, background=0
- Multi-channel images have shape `(C, H, W)` or `(Z, C, H, W)`; channel axis is **not** last
- `pixelsize` is always in µm and passed explicitly — never hardcoded

### Constants

Module-level constants use `UPPER_SNAKE_CASE`. Feature name lists (`AVAILABLE_MASK_FEATURES`, `AVAILABLE_IMAGE_FEATURES`) define the valid string keys accepted by feature-computation functions; extend these lists when adding new features.

### Comments

Comments are used sparingly — only for non-obvious algorithm steps, grouping logically related operations, or brief labels on complex numerical expressions. Do not describe what the code does if the function and variable names already make it clear.
