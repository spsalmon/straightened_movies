# Straightened Movies Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ori_align.py` and `movie_generation.py` with one `straightened_movies.py` built on `towbintools` / `towbintools_pipeline`, adding selectable `center` / `left` frame alignment.

**Architecture:** One script in four sections — generic helpers written in towbintools house style (liftable upstream verbatim), orientation estimation, movie assembly, and pipeline-block plumbing. Movie geometry moves from pad-roll-crop to direct frame placement, which makes alignment a single anchor vector. The block produces no filemap output; the orientation table is a re-run cache only.

**Tech Stack:** Python 3.12, numpy, polars, scikit-image, joblib, tifffile, pytest. `towbintools` 0.4.3, `towbintools_pipeline` 0.1.0.

**Spec:** `docs/superpowers/specs/2026-09-14-straightened-movies-rewrite-design.md`

## Global Constraints

- Every command runs through the environment: `~/.local/bin/micromamba run -n towbintools <command>`. Never bare `python` / `pip` / `pytest`.
- Style: black line length 88, isort `profile=black` + `force_single_line=true`, flake8 `--max-line-length=88 --extend-ignore=E501 E203`, pyupgrade `--py39-plus`.
- Type annotations on every public function. py39+ lowercase generics (`list[str]`, `tuple[int, int]`). `T | None` for optionals. `np.ndarray` for images.
- Docstrings in the towbintools format: one-sentence summary, optional elaboration, then `Parameters:` with `name (type): description` and `(default: value)` noted inline, `Returns:`, and `Raises:` only when the function raises. Double backticks for inline code.
- Image shapes are `(..., H, W)`. The head-tail (length) axis is `-1`; the dorsoventral axis is `-2`.
- No one-line wrapper functions. No docstrings restating a self-evident signature. Comments only for non-obvious steps.
- Commit messages: imperative mood, `feat:` / `fix:` / `refactor:` / `test:` / `docs:` prefix, body explains why. **Never mention AI tools, assistants, or Claude in commit messages.**
- The reference experiment used for real-data checks is
  `/mnt/towbin.data/shared/kstojanovski/20240202_Orca_10x_yap-1del_col-10-tir_wBT160-186-310-337-380-393_25C_20240202_171239_051`
  with filemap `analysis_sacha/report/analysis_filemap.csv`, straightened column
  `analysis_sacha/ch1_ch2_raw_str`, QC column `ch2_seg_str_worm_type`, orientation
  channel 1, and atlas `/mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv`
  with atlas channel 1. **Read-only. Never write into it.**

---

## File Structure

| File | Responsibility |
|---|---|
| `straightened_movies.py` | The whole block: generic helpers, orientation estimation, movie assembly, CLI/worker plumbing |
| `tests/test_straightened_movies.py` | Full test suite |
| `tests/README.md` | How to run the tests |
| `run_straightened_movies.slurm` | Standalone SLURM runner |
| `configs/straightened_movies.yaml` | Example pipeline config using the custom block |
| `README.md` | Usage, replacing `README.txt` |
| `legacy/` | The superseded scripts, kept for reference |
| `~/towbintools_pipeline` (branch) | The no-output block type |

`straightened_movies.py` is one file by explicit request. It stays navigable through four banner comments; the generic-helper section has no dependency on anything below it, which is what makes it liftable into `towbintools` later.

---

### Task 1: Repository setup and legacy archive

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `.pre-commit-config.yaml`, `tests/README.md`, `legacy/README.md`
- Move: `ori_align.py`, `movie_generation.py`, `_debug_movie_generation.py`, `run_ori_prediction.slurm`, `run_movie_generation.slurm`, `_run_debug_movie_generation.slurm`, `README.txt`, `example_output.csv` → `legacy/`
- Delete: `__pycache__/`

**Interfaces:**
- Consumes: nothing
- Produces: a git repository at `/home/spsalmon/straightened_movies` with the legacy scripts preserved under `legacy/`, and a `tests/` directory pytest can collect.

- [ ] **Step 1: Initialise the repository**

```bash
cd /home/spsalmon/straightened_movies
git init
git config user.email "psalmonsacha@gmail.com"
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: Write `pyproject.toml`**

Only tool configuration; this folder is a script, not a distributed package.

```toml
[tool.black]
target-version = ["py311"]

[tool.isort]
profile = "black"
force_single_line = true
```

- [ ] **Step 4: Write `.pre-commit-config.yaml`**

Copy the hook set from `towbintools_pipeline/.pre-commit-config.yaml`:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
    -   id: check-yaml
    -   id: debug-statements
    -   id: end-of-file-fixer
    -   id: trailing-whitespace
-   repo: https://github.com/PyCQA/isort
    rev: 8.0.1
    hooks:
    -   id: isort
-   repo: https://github.com/asottile/pyupgrade
    rev: v3.21.2
    hooks:
    -   id: pyupgrade
        args: [--py39-plus]
-   repo: https://github.com/psf/black
    rev: 26.3.1
    hooks:
    -   id: black
-   repo: https://github.com/PyCQA/flake8
    rev: 7.3.0
    hooks:
    -   id: flake8
        args: [--max-line-length=88, --extend-ignore=E501 E203]
```

- [ ] **Step 5: Archive the superseded scripts**

```bash
cd /home/spsalmon/straightened_movies
rm -rf __pycache__
mkdir -p legacy configs tests
git mv 2>/dev/null || true
mv ori_align.py movie_generation.py _debug_movie_generation.py \
   run_ori_prediction.slurm run_movie_generation.slurm \
   _run_debug_movie_generation.slurm README.txt example_output.csv legacy/
```

- [ ] **Step 6: Write `legacy/README.md`**

```markdown
# Superseded scripts

These are the original orientation-prediction and movie-generation scripts,
kept for reference and for reproducing results produced before the rewrite.
They are not maintained. Use `../straightened_movies.py` instead.

- `ori_align.py` — orientation prediction, wrote `orientations.csv`
- `movie_generation.py` — movie assembly from `orientations.csv`
- `_debug_movie_generation.py` — debugging variant of the above
- `example_output.csv` — a real `orientations.csv` from the 20240202 experiment,
  used to check that registration shifts are jitter rather than drift
- `README.txt` — the original usage notes
```

- [ ] **Step 7: Write `tests/README.md`**

```markdown
# Tests

`test_straightened_movies.py` covers the geometry helpers, the flip-transform
algebra, orientation prediction and movie assembly, plus an end-to-end run of
the script as a subprocess against a synthetic experiment generated at runtime.
No bundled data, no SLURM, no pipeline required.

Run from the repository root:

```bash
~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v
```

The tests skip themselves if the dependencies are not installed.
```

- [ ] **Step 8: Verify pytest collects cleanly**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: `no tests ran`, exit code 5, no collection errors.

- [ ] **Step 9: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add -A
git commit -m "chore: initialise repository and archive the pre-rewrite scripts

The orientation and movie scripts predate towbintools and towbintools_pipeline
and are being replaced by a single block. Keep them under legacy/ so results
produced with them stay reproducible."
```

---

### Task 2: Frame placement geometry

**Files:**
- Create: `straightened_movies.py`
- Test: `tests/test_straightened_movies.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ALIGNMENTS: dict[str, tuple[float, float]]` — `{"center": (0.5, 0.5), "left": (0.5, 0.0)}`
  - `registration_origins(shapes, shifts, anchors=(0.5, 0.5), center=False) -> tuple[np.ndarray, tuple[int, int]]`
  - `assemble_registered_stack(images, origins, canvas_shape, fill=0) -> np.ndarray`
  - `crop_to_mask(image, mask) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_straightened_movies.py`:

```python
"""Tests for straightened_movies.py.

Synthetic data is generated at runtime; nothing here needs a real experiment,
SLURM, or a pipeline run.
"""

import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("skimage")
pytest.importorskip("tifffile")
pytest.importorskip("polars")
pytest.importorskip("towbintools")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import straightened_movies as sm  # noqa: E402


# ---- Frame placement geometry ----


def test_registration_origins_left_pins_the_length_axis_to_zero():
    shapes = np.array([[10, 20], [10, 30], [10, 40]])
    shifts = np.array([[0, 3], [0, -2], [0, 5]])

    origins, canvas = sm.registration_origins(
        shapes, shifts, anchors=sm.ALIGNMENTS["left"]
    )

    # The frame with the smallest shift defines x = 0 on the length axis.
    assert origins[:, 1].min() == 0
    assert origins[1, 1] == 0
    # The canvas is the tight union of the placed frames.
    assert canvas == (10, 47)


def test_registration_origins_preserves_relative_spacing_under_every_anchor():
    shapes = np.array([[10, 20], [10, 20], [10, 20]])
    shifts = np.array([[0, 3], [0, -2], [0, 5]])

    centered, _ = sm.registration_origins(
        shapes, shifts, anchors=sm.ALIGNMENTS["center"]
    )
    left, _ = sm.registration_origins(shapes, shifts, anchors=sm.ALIGNMENTS["left"])

    # Alignment moves every frame by the same amount; the registration between
    # frames is identical either way.
    assert np.array_equal(np.diff(centered, axis=0), np.diff(left, axis=0))
    assert np.array_equal(np.diff(centered[:, 1]), np.diff(shifts[:, 1]))


def test_registration_origins_center_balances_margins_for_equal_frames():
    shapes = np.array([[10, 20], [10, 20]])
    shifts = np.array([[0, 0], [0, 0]])

    origins, canvas = sm.registration_origins(
        shapes, shifts, anchors=sm.ALIGNMENTS["center"]
    )

    assert np.array_equal(origins, np.zeros((2, 2), dtype=int))
    assert canvas == (10, 20)


def test_registration_origins_center_mode_keeps_the_origin_at_the_canvas_centre():
    shapes = np.array([[10, 20], [10, 30]])
    shifts = np.array([[0, 4], [0, -6]])

    origins, canvas = sm.registration_origins(
        shapes, shifts, anchors=sm.ALIGNMENTS["center"], center=True
    )

    # A frame whose anchored position is p sits at p + canvas/2, so the common
    # registration origin lands exactly in the middle of the canvas.
    anchored = shifts - np.rint(np.array([0.5, 0.5]) * shapes).astype(int)
    assert np.array_equal(origins - anchored, np.array(canvas) // 2)
    assert canvas[0] % 2 == 0 and canvas[1] % 2 == 0
    assert (origins >= 0).all()
    assert ((origins + shapes) <= np.array(canvas)).all()


def test_assemble_registered_stack_places_each_frame_at_its_origin():
    images = [np.full((2, 3), 1, dtype=np.uint16), np.full((2, 2), 2, dtype=np.uint16)]
    origins = np.array([[0, 0], [1, 2]])

    stack = sm.assemble_registered_stack(images, origins, (3, 4))

    assert stack.shape == (2, 3, 4)
    assert stack.dtype == np.uint16
    assert np.array_equal(stack[0, 0:2, 0:3], images[0])
    assert np.array_equal(stack[1, 1:3, 2:4], images[1])
    assert stack[1, 0, 0] == 0


def test_assemble_registered_stack_keeps_the_channel_axis():
    images = [np.ones((3, 2, 2), dtype=np.uint8), np.ones((3, 2, 2), dtype=np.uint8)]
    origins = np.array([[0, 0], [1, 1]])

    stack = sm.assemble_registered_stack(images, origins, (3, 3))

    assert stack.shape == (2, 3, 3, 3)


def test_assemble_registered_stack_rejects_mismatched_leading_dimensions():
    images = [np.ones((3, 2, 2)), np.ones((2, 2, 2))]

    with pytest.raises(ValueError, match="leading dimensions"):
        sm.assemble_registered_stack(images, np.zeros((2, 2), dtype=int), (2, 2))


def test_crop_to_mask_returns_the_bounding_box():
    image = np.arange(20).reshape(4, 5)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True

    assert np.array_equal(sm.crop_to_mask(image, mask), image[1:3, 2:4])


def test_crop_to_mask_rejects_an_empty_mask():
    with pytest.raises(ValueError, match="empty mask"):
        sm.crop_to_mask(np.zeros((4, 5)), np.zeros((4, 5), dtype=bool))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'straightened_movies'`.

- [ ] **Step 3: Write the module header and the geometry helpers**

Create `straightened_movies.py`:

```python
"""Orientation prediction and straightened-movie assembly for one experiment.

For every point, the straightened images classified as worms are aligned to one
another, given an absolute orientation by comparison against a manually oriented
atlas, and assembled into a movie with the frames registered to remove jitter.

Runs standalone or as a ``towbintools_pipeline`` custom building block. It
produces no filemap-visible output: movies are written under ``--movie_dir`` and
the orientation table under ``--orientations``, the latter purely as a cache that
makes re-runs over the same experiment cheap.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Generic helpers -- candidates for towbintools
#
# Nothing in this section depends on anything below it, so it can be lifted into
# towbintools as-is.
# ---------------------------------------------------------------------------

# Anchor position per axis, as (dorsoventral, head-tail), from 0.0 at the
# low-index edge to 1.0 at the high-index edge. "left" holds the head end of
# every frame against the left of the canvas so that the worm only grows
# rightwards; "center" lets it grow in both directions.
ALIGNMENTS = {
    "center": (0.5, 0.5),
    "left": (0.5, 0.0),
}


def registration_origins(
    shapes: np.ndarray,
    shifts: np.ndarray,
    anchors: tuple[float, float] = (0.5, 0.5),
    center: bool = False,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Place registered frames of differing sizes into a shared canvas.

    Each frame is anchored within the canvas according to ``anchors`` and then
    translated by its registration shift. A frame anchored at ``a`` within a canvas
    of size ``T`` starts at ``a * (T - shape)``, so its origin is
    ``shift + a * T - a * shape``; the ``a * T`` term is common to every frame and
    drops out once the origins are rebased. The difference between two frames'
    origins is therefore their difference in shift whatever the anchor, which is
    what keeps the frames registered under every alignment.

    Parameters:
        shapes (np.ndarray): Size of each frame along the two anchored axes, of
            shape ``(N, 2)``.
        shifts (np.ndarray): Registration shift of each frame, of shape ``(N, 2)``.
        anchors (tuple[float, float]): Anchor position per axis, from ``0.0`` at the
            low-index edge to ``1.0`` at the high-index edge. (default: (0.5, 0.5))
        center (bool): If ``True``, grow the canvas symmetrically about the common
            registration origin instead of cropping it to the frames. Needed when the
            assembled stack is averaged and measured again by phase cross correlation,
            which reports shifts relative to image centres. (default: False)

    Returns:
        tuple[np.ndarray, tuple[int, int]]: The origin of each frame, of shape
            ``(N, 2)``, and the size of the canvas holding every frame.
    """
    shapes = np.asarray(shapes, dtype=int)
    shifts = np.asarray(shifts, dtype=int)
    anchored = shifts - np.rint(np.asarray(anchors) * shapes).astype(int)

    if center:
        half = np.maximum(-anchored.min(axis=0), (anchored + shapes).max(axis=0))
        origins = anchored + half
        canvas = 2 * half
    else:
        origins = anchored - anchored.min(axis=0)
        canvas = (origins + shapes).max(axis=0)

    return origins, tuple(int(size) for size in canvas)


def assemble_registered_stack(
    images: list[np.ndarray],
    origins: np.ndarray,
    canvas_shape: tuple[int, int],
    fill: float = 0,
) -> np.ndarray:
    """
    Paste registered frames into a common canvas, stacked along a leading axis.

    Parameters:
        images (list[np.ndarray]): Frames of shape ``(..., H, W)``, all sharing the
            same leading dimensions.
        origins (np.ndarray): Origin of each frame on the two trailing axes, of shape
            ``(N, 2)``, as returned by ``registration_origins``.
        canvas_shape (tuple[int, int]): Size of the canvas' two trailing axes.
        fill (float): Value the canvas is initialised with. (default: 0)

    Returns:
        np.ndarray: The stack, of shape ``(N, ..., canvas_shape[0], canvas_shape[1])``.

    Raises:
        ValueError: If the frames do not all share the same leading dimensions.
    """
    leading = images[0].shape[:-2]
    if any(image.shape[:-2] != leading for image in images):
        raise ValueError(
            "All frames must share the same leading dimensions, got "
            f"{sorted({image.shape[:-2] for image in images})}"
        )

    stack = np.full((len(images), *leading, *canvas_shape), fill, dtype=images[0].dtype)
    for index, (image, origin) in enumerate(zip(images, origins)):
        rows = slice(origin[0], origin[0] + image.shape[-2])
        columns = slice(origin[1], origin[1] + image.shape[-1])
        stack[index, ..., rows, columns] = image
    return stack


def crop_to_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop an image to the bounding box of a boolean mask.

    Parameters:
        image (np.ndarray): Image of shape ``(..., H, W)``.
        mask (np.ndarray): Boolean mask of shape ``(H, W)``.

    Returns:
        np.ndarray: The image cropped to the mask's bounding box.

    Raises:
        ValueError: If the mask does not match the image's trailing axes, or is empty.
    """
    if mask.shape != image.shape[-2:]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match the image's trailing axes "
            f"{image.shape[-2:]}"
        )
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("Cannot crop to an empty mask")
    return image[..., rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add straightened_movies.py tests/test_straightened_movies.py
git commit -m "feat: place movie frames by computed origin instead of pad-roll-crop

The old assembly padded every frame into a canvas of twice the largest frame,
rolled it by its shift, then cropped the black borders back off using a parallel
stack of boolean masks, with a flush_axes argument threaded through all three
steps to support anything but centring. Computing each frame's origin directly
makes the canvas the union bounding box by construction, removes the wrap-around
risk of np.roll, and reduces alignment to one anchor vector."
```

---

### Task 3: Intensity normalisation, similarity and translation

**Files:**
- Modify: `straightened_movies.py`
- Test: `tests/test_straightened_movies.py`

**Interfaces:**
- Consumes: `crop_to_mask` (Task 2)
- Produces:
  - `normalize_images_to_common_range(images, out_range="float32") -> list[np.ndarray]`
  - `structural_dissimilarity(image, other, channel_axis=None) -> float`
  - `estimate_translation(reference, moving) -> tuple[np.ndarray, float]`

**Note on `estimate_translation`:** the original `optimal_translation` called
`np.roll(mov_padded, shift)` with no `axis` argument, which flattens the array and
rolls it by `dy + dx` rather than rolling each axis. The port below passes
`axis=(-2, -1)`. Task 8 measures both variants on real data and settles which one
ships; until then the fixed version is the default because it is what the
surrounding code intends.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_straightened_movies.py`:

```python
# ---- Normalisation, similarity, translation ----


def test_normalize_images_to_common_range_uses_one_range_for_the_series():
    images = [
        np.array([[0, 50]], dtype=np.uint16),
        np.array([[50, 100]], dtype=np.uint16),
    ]

    normalized = sm.normalize_images_to_common_range(images)

    # A shared range means the 50 in both images maps to the same value, which a
    # per-image normalisation would not do.
    assert normalized[0][0, 1] == pytest.approx(normalized[1][0, 0])
    assert normalized[0][0, 0] == pytest.approx(0.0)
    assert normalized[1][0, 1] == pytest.approx(1.0)


def test_normalize_images_to_common_range_keeps_dimensionality():
    two_d = [np.ones((4, 5), dtype=np.uint16), np.zeros((4, 5), dtype=np.uint16)]
    three_d = [np.ones((2, 4, 5), dtype=np.uint16), np.zeros((2, 4, 5), dtype=np.uint16)]

    assert sm.normalize_images_to_common_range(two_d)[0].shape == (4, 5)
    assert sm.normalize_images_to_common_range(three_d)[0].shape == (2, 4, 5)


def test_structural_dissimilarity_is_zero_for_identical_images():
    rng = np.random.default_rng(0)
    image = rng.random((40, 40)).astype(np.float32)

    assert sm.structural_dissimilarity(image, image) == pytest.approx(0.0, abs=1e-6)


def test_structural_dissimilarity_grows_with_difference():
    rng = np.random.default_rng(0)
    image = rng.random((40, 40)).astype(np.float32)
    slightly_off = image + rng.normal(0, 0.01, image.shape).astype(np.float32)
    very_off = rng.random((40, 40)).astype(np.float32)

    assert sm.structural_dissimilarity(image, slightly_off) < sm.structural_dissimilarity(
        image, very_off
    )


def test_structural_dissimilarity_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        sm.structural_dissimilarity(np.zeros((4, 4)), np.zeros((4, 5)))


def _blob(height=60, width=80, row=20, column=10):
    # An asymmetric bright patch: wider than tall and offset, so that every flip
    # and translation of it is distinguishable from the original.
    image = np.zeros((height, width), dtype=np.float32)
    image[row : row + 12, column : column + 30] = 1.0
    image[row : row + 4, column : column + 6] = 0.4
    return image


def test_estimate_translation_recovers_a_known_shift():
    reference = _blob()
    moving = np.roll(reference, (5, -7), axis=(0, 1))

    shift, dissimilarity = sm.estimate_translation(reference, moving)

    assert tuple(shift) == (-5, 7)
    assert dissimilarity == pytest.approx(0.0, abs=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v -k "normalize or dissimilarity or translation"`
Expected: FAIL — `AttributeError: module 'straightened_movies' has no attribute 'normalize_images_to_common_range'`.

- [ ] **Step 3: Add the imports**

At the top of `straightened_movies.py`, after the docstring, replace the lone `import numpy as np` with:

```python
import numpy as np
from skimage import exposure
from skimage import metrics
from skimage import registration
from towbintools.foundation.image_handling import pad_to_dim_equally
```

- [ ] **Step 4: Implement the three functions**

Append to the generic-helpers section of `straightened_movies.py`, after `crop_to_mask`:

```python
def normalize_images_to_common_range(
    images: list[np.ndarray],
    out_range: str = "float32",
) -> list[np.ndarray]:
    """
    Rescale a series of images against an intensity range shared by the whole series.

    Every channel is rescaled from its minimum and maximum across all the images, so
    that frames stay comparable to one another. ``image_handling.normalize_image``
    rescales a single image and would give every frame a range of its own.

    Parameters:
        images (list[np.ndarray]): Images of shape ``(C, H, W)`` or ``(H, W)``, all
            with the same number of channels.
        out_range (str): Output range, passed to
            ``skimage.exposure.rescale_intensity``. (default: "float32")

    Returns:
        list[np.ndarray]: The rescaled images, keeping the dimensionality of the input.
    """
    stacks = [image[np.newaxis, ...] if image.ndim == 2 else image for image in images]
    channel_ranges = [
        (
            min(stack[channel].min() for stack in stacks),
            max(stack[channel].max() for stack in stacks),
        )
        for channel in range(stacks[0].shape[0])
    ]
    return [
        np.stack(
            [
                exposure.rescale_intensity(
                    channel, in_range=channel_range, out_range=out_range
                )
                for channel, channel_range in zip(stack, channel_ranges)
            ]
        ).squeeze()
        for stack in stacks
    ]


def structural_dissimilarity(
    image: np.ndarray,
    other: np.ndarray,
    channel_axis: int | None = None,
) -> float:
    """
    Measure how dissimilar two images of the same shape are, as ``1 - SSIM``.

    The comparison window is 21 pixels rather than scikit-image's default of 7. The
    longer range is more forgiving of large morphological differences and of slight
    channel misalignment, which widens the dynamic range of the measure and makes two
    candidate alignments easier to tell apart.

    Parameters:
        image (np.ndarray): First image, of shape ``(..., H, W)``.
        other (np.ndarray): Second image, of the same shape.
        channel_axis (int | None): Axis holding channels, or ``None`` for a
            single-channel image. (default: None)

    Returns:
        float: The dissimilarity, ``0.0`` for identical images and larger the more
            they differ.

    Raises:
        ValueError: If the two images do not have the same shape.
    """
    if image.shape != other.shape:
        raise ValueError(
            f"Images must have the same shape, got {image.shape} and {other.shape}"
        )
    window = min(21, *image.shape)
    # structural_similarity requires an odd window.
    window = window - ((window + 1) % 2)
    data_range = max(image.max(), other.max()) - min(image.min(), other.min())
    similarity = metrics.structural_similarity(
        image,
        other,
        data_range=data_range,
        win_size=window,
        channel_axis=channel_axis,
        gaussian_weights=False,
        use_sample_covariance=False,
    )
    return 1 - similarity


def estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Find the translation that best registers one image onto another.

    Both images are padded to the sum of their shapes so that the circular
    convolution behind phase cross correlation cannot wrap one edge onto the other,
    and the dissimilarity is measured over the region where they actually overlap.

    Parameters:
        reference (np.ndarray): The image being registered onto, of shape ``(H, W)``.
        moving (np.ndarray): The image being moved, of shape ``(H, W)``.

    Returns:
        tuple[np.ndarray, float]: The integer shift applying ``moving`` onto
            ``reference``, and the dissimilarity of the two once registered.
    """
    target = np.array(reference.shape) + np.array(moving.shape) - 1
    reference_padded = pad_to_dim_equally(reference, *target)
    reference_mask = pad_to_dim_equally(np.ones_like(reference, dtype=bool), *target)
    moving_padded = pad_to_dim_equally(moving, *target)
    moving_mask = pad_to_dim_equally(np.ones_like(moving, dtype=bool), *target)

    # The masked variant always returns an integer shift, but types it as a float.
    shift, _, _ = registration.phase_cross_correlation(
        reference_image=reference_padded,
        reference_mask=reference_mask,
        moving_image=moving_padded,
        moving_mask=moving_mask,
        overlap_ratio=0.9,
        normalization=None,
    )
    shift = shift.astype(int)

    moving_padded = np.roll(moving_padded, shift, axis=(-2, -1))
    moving_mask = np.roll(moving_mask, shift, axis=(-2, -1))

    overlap = reference_mask & moving_mask
    dissimilarity = structural_dissimilarity(
        crop_to_mask(reference_padded, overlap),
        crop_to_mask(moving_padded, overlap),
    )
    return shift, dissimilarity
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: 15 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add straightened_movies.py tests/test_straightened_movies.py
git commit -m "feat: port intensity normalisation, SSIM scoring and translation

estimate_translation now rolls the moving image along its two spatial axes.
The original passed no axis to np.roll, which flattens the array and rolls it by
dy + dx, so the dissimilarity that ranks candidate flips was measured on a
misaligned image."
```

---

### Task 4: Flip transforms

**Files:**
- Modify: `straightened_movies.py`
- Test: `tests/test_straightened_movies.py`

**Interfaces:**
- Consumes: `estimate_translation` (Task 3)
- Produces:
  - `class FlipTransform` with `__init__(mirror_axes=None)`, `__call__(image)`, `__add__`, `__hash__`, `__eq__`, `__contains__`, `__str__`, `__repr__`, `orientation_labels() -> dict[str, str]`, and `all_transforms(ndim, channel_axis=None) -> list[FlipTransform]` as a `staticmethod`
  - `estimate_flip_transform(reference, moving, channel_axis=None, scale=None) -> tuple[FlipTransform, np.ndarray, float]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_straightened_movies.py`:

```python
# ---- Flip transforms ----


def test_flip_transform_composition_is_symmetric_difference():
    flip_rows = sm.FlipTransform((0,))
    flip_both = sm.FlipTransform((0, 1))

    assert flip_rows + flip_both == sm.FlipTransform((1,))
    # A flip is its own inverse.
    assert flip_rows + flip_rows == sm.FlipTransform()


def test_flip_transform_is_hashable_and_compares_by_axes():
    assert sm.FlipTransform((0, 1)) == sm.FlipTransform((1, 0))
    assert len({sm.FlipTransform((0, 1)), sm.FlipTransform((1, 0))}) == 1


def test_flip_transform_orientation_labels():
    # Head left and vulva up is the atlas convention, so an identity transform
    # reports the worm as already facing that way.
    assert sm.FlipTransform().orientation_labels() == {"head": "L", "vulva": "U"}
    assert sm.FlipTransform((1,)).orientation_labels() == {"head": "R", "vulva": "U"}
    assert sm.FlipTransform((0,)).orientation_labels() == {"head": "L", "vulva": "D"}
    assert sm.FlipTransform((0, 1)).orientation_labels() == {"head": "R", "vulva": "D"}


def test_flip_transform_applies_the_flip():
    image = np.array([[1, 2], [3, 4]])

    assert np.array_equal(sm.FlipTransform((1,))(image), np.array([[2, 1], [4, 3]]))
    assert np.array_equal(sm.FlipTransform()(image), image)


def test_all_transforms_enumerates_every_combination():
    assert len(sm.FlipTransform.all_transforms(2)) == 4
    assert len(sm.FlipTransform.all_transforms(3)) == 8
    assert len(sm.FlipTransform.all_transforms(3, channel_axis=0)) == 4


def test_estimate_flip_transform_recovers_a_flip_and_a_shift():
    reference = _blob()
    moving = np.flip(reference, axis=1)

    transform, shift, error = sm.estimate_flip_transform(reference, moving)

    assert transform == sm.FlipTransform((1,))
    assert error == pytest.approx(0.0, abs=1e-3)
    assert np.array_equal(shift, np.zeros(2, dtype=int))


def test_estimate_flip_transform_prefers_the_identity_for_an_unflipped_image():
    reference = _blob()
    moving = np.roll(reference, 3, axis=1)

    transform, _, _ = sm.estimate_flip_transform(reference, moving)

    assert transform == sm.FlipTransform()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v -k "flip or transform"`
Expected: FAIL — `AttributeError: module 'straightened_movies' has no attribute 'FlipTransform'`.

- [ ] **Step 3: Add the imports**

Add to the import block of `straightened_movies.py`:

```python
from itertools import compress
from itertools import product

from skimage import transform as sk_transform
```

- [ ] **Step 4: Implement `FlipTransform`**

Append to the generic-helpers section:

```python
class FlipTransform:
    """
    A combination of axis mirrorings, composable with other flips.

    Orientations are expressed relative to the atlas convention of head left and
    vulva up: a transform that mirrors the last axis is the one that turns a
    head-right worm into a head-left one, and therefore also the one that reports
    the original as head-right.
    """

    def __init__(self, mirror_axes: tuple[int, ...] | None = None):
        # A frozenset so that transforms can be dictionary keys when tallying the
        # atlas consensus.
        self.mirror_axes = frozenset(mirror_axes or ())

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.flip(image, axis=tuple(self.mirror_axes))

    def __add__(self, other: "FlipTransform") -> "FlipTransform":
        # Flipping the same axis twice is the identity, so composition is the
        # symmetric difference of the mirrored axes.
        return FlipTransform(tuple(self.mirror_axes ^ other.mirror_axes))

    def __hash__(self) -> int:
        return hash(self.mirror_axes)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FlipTransform) and self.mirror_axes == other.mirror_axes
        )

    def __contains__(self, axis: int) -> bool:
        return axis in self.mirror_axes

    def __repr__(self) -> str:
        return f"FlipTransform(mirror_axes={tuple(sorted(self.mirror_axes))})"

    def orientation_labels(self) -> dict[str, str]:
        """
        Report the orientation of the image this transform aligns to the atlas.

        Returns:
            dict[str, str]: ``"head"`` as ``"L"`` or ``"R"`` and ``"vulva"`` as
                ``"U"`` or ``"D"``.
        """
        return {
            "head": "R" if 1 in self else "L",
            "vulva": "D" if 0 in self else "U",
        }

    @staticmethod
    def all_transforms(
        ndim: int, channel_axis: int | None = None
    ) -> list["FlipTransform"]:
        """
        Enumerate every combination of mirrorings over an image's axes.

        Parameters:
            ndim (int): Number of axes of the images being transformed.
            channel_axis (int | None): Axis to leave alone, or ``None`` to mirror
                every axis. (default: None)

        Returns:
            list[FlipTransform]: The ``2 ** n`` transforms over the mirrored axes.
        """
        axes = list(range(ndim))
        if channel_axis is not None:
            axes.pop(channel_axis)
        return [
            FlipTransform(tuple(compress(axes, included)))
            for included in product((False, True), repeat=len(axes))
        ]
```

- [ ] **Step 5: Implement `estimate_flip_transform`**

Append to the generic-helpers section:

```python
def estimate_flip_transform(
    reference: np.ndarray,
    moving: np.ndarray,
    channel_axis: int | None = None,
    scale: float | None = None,
) -> tuple["FlipTransform", np.ndarray, float]:
    """
    Find the mirroring that best aligns one image onto another.

    Every combination of mirrorings is registered onto the reference by phase cross
    correlation, and scored by how dissimilar the registered pair is plus a penalty
    on how far the image had to move. The penalty ignores the movement needed to
    account for the two images simply being different sizes, so that a large but
    necessary shift is not mistaken for a poor match.

    Parameters:
        reference (np.ndarray): The image being aligned onto.
        moving (np.ndarray): The image being mirrored and moved.
        channel_axis (int | None): Axis holding channels, left unmirrored.
            (default: None)
        scale (float | None): Factor below 1 to rescale both images by before
            searching, trading a little accuracy for speed on large images. The
            returned shift is expressed at the original scale. (default: None)

    Returns:
        tuple[FlipTransform, np.ndarray, float]: The best transform, the integer
            shift registering the mirrored image onto the reference, and the score.
    """
    if scale is not None and scale < 1:
        reference = sk_transform.rescale(
            reference, scale, anti_aliasing=True, channel_axis=channel_axis
        )
        moving = sk_transform.rescale(
            moving, scale, anti_aliasing=True, channel_axis=channel_axis
        )

    size_mismatch = np.round(
        np.abs(np.array(reference.shape) - np.array(moving.shape)) / 2
    )
    largest_axis = np.max(np.vstack([reference.shape, moving.shape]), axis=0).max()

    candidates = FlipTransform.all_transforms(reference.ndim, channel_axis=channel_axis)
    shifts = []
    errors = []
    for candidate in candidates:
        shift, dissimilarity = estimate_translation(reference, candidate(moving))
        movement = np.clip(np.abs(shift) - size_mismatch, 0, None)
        errors.append(dissimilarity + (movement / largest_axis).sum() * 2)
        shifts.append(shift)

    best = int(np.argmin(errors))
    shift = shifts[best]
    if scale is not None and scale < 1:
        shift = np.round(shift / scale).astype(int)
    return candidates[best], shift, errors[best]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: 22 passed.

- [ ] **Step 7: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add straightened_movies.py tests/test_straightened_movies.py
git commit -m "feat: port the flip-transform search

Same scoring as before: SSIM dissimilarity plus a movement penalty that ignores
the shift needed to account for a size difference between the two images. The
estimate_transform classmethod becomes a module-level function, and the
self-inverse 'inverse' property is dropped since it returned the transform."
```

---

### Task 5: Orientation estimation

**Files:**
- Modify: `straightened_movies.py`
- Test: `tests/test_straightened_movies.py`

**Interfaces:**
- Consumes: `registration_origins`, `assemble_registered_stack` (Task 2), `normalize_images_to_common_range` (Task 3), `FlipTransform`, `estimate_flip_transform` (Task 4)
- Produces:
  - `RUNNING_MEAN_WINDOW: int = 5`, `MAX_REGISTRATION_PIXELS: int = 40_000`, `ATLAS_QUERIES: int = 50`
  - `mean_registered_image(images, shifts) -> np.ndarray`
  - `closest_shape_match(image, references) -> np.ndarray`
  - `running_mean_orientation(images, window=RUNNING_MEAN_WINDOW, channel_axis=None) -> tuple[list[np.ndarray], list[FlipTransform], np.ndarray]`
  - `align_to_atlas(images, atlas, n_queries=ATLAS_QUERIES) -> tuple[FlipTransform, float]`
  - `predict_orientations(images, atlas) -> tuple[np.ndarray, list[dict[str, str]], float]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_straightened_movies.py`:

```python
# ---- Orientation estimation ----


def _worm(length, height=24, flip=None):
    # A straightened-worm proxy: a bright "pharynx" near the head end and a dimmer
    # body, so that head and tail are distinguishable and the image is asymmetric
    # along both axes.
    image = np.zeros((height, length), dtype=np.float32)
    image[6:18, :] = 0.3
    image[6:18, : length // 5] = 1.0
    image[6:10, length // 3 : length // 2] = 0.7
    if flip is not None:
        image = np.flip(image, axis=flip)
    return image


def test_closest_shape_match_picks_the_nearest_reference():
    references = [np.zeros((10, 10)), np.zeros((10, 40)), np.zeros((10, 100))]

    match = sm.closest_shape_match(np.zeros((10, 45)), references)

    assert match.shape == (10, 40)


def test_mean_registered_image_averages_onto_a_common_frame():
    images = [np.ones((4, 6), dtype=np.float32)] * 3
    shifts = np.zeros((3, 2), dtype=int)

    mean = sm.mean_registered_image(images, shifts)

    # Centring keeps the registration origin in the middle, so the canvas is
    # symmetric and the frames land on top of one another.
    assert mean.shape[0] % 2 == 0 and mean.shape[1] % 2 == 0
    assert mean.max() == pytest.approx(1.0)


def test_running_mean_orientation_flips_a_series_into_agreement():
    lengths = [40, 44, 48, 52, 56, 60, 64, 68]
    images = [_worm(length) for length in lengths]
    # Flip half the series along the length axis; the running mean should undo it.
    images[4:] = [np.flip(image, axis=1) for image in images[4:]]

    oriented, transforms, shifts = sm.running_mean_orientation(images, window=3)

    assert len(oriented) == len(transforms) == len(images)
    assert shifts.shape == (len(images), 2)
    # Whatever absolute direction it settles on, the series must be self-consistent:
    # the bright pharynx ends up on the same side in every frame.
    sides = [image[:, : image.shape[1] // 2].sum() > image[:, image.shape[1] // 2 :].sum()
             for image in oriented]
    assert len(set(sides)) == 1


def test_align_to_atlas_reports_the_flip_and_full_agreement():
    atlas = [_worm(length) for length in (40, 50, 60, 70)]
    images = [np.flip(_worm(length), axis=1) for length in (42, 52, 62)]

    transform, confidence = sm.align_to_atlas(images, atlas)

    assert transform == sm.FlipTransform((1,))
    assert confidence == pytest.approx(1.0)


def test_predict_orientations_labels_a_flipped_series():
    atlas = [_worm(length) for length in (40, 50, 60, 70)]
    images = [np.flip(_worm(length), axis=1) for length in (42, 46, 50, 54, 58)]

    shifts, orientations, confidence = sm.predict_orientations(images, atlas)

    assert shifts.shape == (len(images), 2)
    assert all(orientation["head"] == "R" for orientation in orientations)
    assert confidence == pytest.approx(1.0)


def test_predict_orientations_labels_an_unflipped_series():
    atlas = [_worm(length) for length in (40, 50, 60, 70)]
    images = [_worm(length) for length in (42, 46, 50, 54, 58)]

    _, orientations, confidence = sm.predict_orientations(images, atlas)

    assert all(orientation["head"] == "L" for orientation in orientations)
    assert confidence == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v -k "orientation or atlas or shape_match or mean_registered"`
Expected: FAIL — `AttributeError: module 'straightened_movies' has no attribute 'closest_shape_match'`.

- [ ] **Step 3: Implement the orientation section**

Append to `straightened_movies.py`, starting a new banner section:

```python
# ---------------------------------------------------------------------------
# Orientation estimation
# ---------------------------------------------------------------------------

# Number of previous frames averaged into the reference each image is oriented
# against. A running mean is steadier than the previous frame alone, which lets a
# single bad frame flip the rest of the series.
RUNNING_MEAN_WINDOW = 5
# Images larger than this are rescaled before the flip search. Registration is
# quadratic in image size and loses very little accuracy at this resolution.
MAX_REGISTRATION_PIXELS = 40_000
# How many frames of a series are compared against the atlas to decide its
# absolute orientation.
ATLAS_QUERIES = 50


def _registration_scale(image: np.ndarray) -> float:
    if image.size < MAX_REGISTRATION_PIXELS:
        return 1.0
    # The scale applies to axis lengths, so size scales with scale ** ndim.
    return (MAX_REGISTRATION_PIXELS / image.size) ** (1 / image.ndim)


def mean_registered_image(
    images: list[np.ndarray],
    shifts: np.ndarray,
) -> np.ndarray:
    """
    Average a set of registered images onto a common frame.

    The canvas is centred on the registration origin rather than cropped to the
    images, because the result is measured again by phase cross correlation, which
    reports shifts relative to image centres.

    Parameters:
        images (list[np.ndarray]): Images of shape ``(H, W)``, of any sizes.
        shifts (np.ndarray): Registration shift of each image, of shape ``(N, 2)``.

    Returns:
        np.ndarray: The mean image.
    """
    shapes = np.array([image.shape[-2:] for image in images])
    origins, canvas = registration_origins(shapes, shifts, center=True)
    stack = assemble_registered_stack(
        [image.astype(np.float32) for image in images], origins, canvas
    )
    return stack.mean(axis=0)


def closest_shape_match(
    image: np.ndarray,
    references: list[np.ndarray],
) -> np.ndarray:
    """
    Pick the reference whose shape is closest to an image's.

    Straightened worms grow along their length, so shape stands in for
    developmental stage and keeps the atlas comparison between worms of a similar
    size.

    Parameters:
        image (np.ndarray): The image to match.
        references (list[np.ndarray]): The images to choose from.

    Returns:
        np.ndarray: The closest reference.
    """
    distances = np.abs(
        np.array([reference.shape for reference in references]) - np.array(image.shape)
    ).sum(axis=1)
    return references[int(np.argmin(distances))]


def running_mean_orientation(
    images: list[np.ndarray],
    window: int = RUNNING_MEAN_WINDOW,
    channel_axis: int | None = None,
) -> tuple[list[np.ndarray], list["FlipTransform"], np.ndarray]:
    """
    Orient a series of images so that every frame faces the same way as the others.

    Each image is compared against the mean of the frames already oriented before it.
    The absolute direction the series settles on is arbitrary at this stage; only
    agreement within the series is established.

    Parameters:
        images (list[np.ndarray]): The series, in time order.
        window (int): Number of previous frames averaged into the reference.
            (default: RUNNING_MEAN_WINDOW)
        channel_axis (int | None): Axis holding channels, left unmirrored.
            (default: None)

    Returns:
        tuple[list[np.ndarray], list[FlipTransform], np.ndarray]: The oriented
            images, the transform applied to each, and the cumulative shifts of
            shape ``(N, 2)``.
    """
    seed = _bootstrap_mean_image(images[:window], channel_axis=channel_axis)

    transforms: list[FlipTransform] = []
    oriented = [seed] * window
    shifts = [np.zeros(2, dtype=int)] * window
    for image in images:
        reference = mean_registered_image(
            oriented[-window:], np.array(shifts[-window:])
        )
        scale = min(_registration_scale(reference), _registration_scale(image))
        transform, shift, _ = estimate_flip_transform(
            reference, image, channel_axis=channel_axis, scale=scale
        )
        transforms.append(transform)
        oriented.append(transform(image))
        shifts.append(shift)

    return oriented[window:], transforms, np.array(shifts[window:])


def _bootstrap_mean_image(
    images: list[np.ndarray],
    channel_axis: int | None = None,
) -> np.ndarray:
    # Seeds the running mean: the first few frames have nothing before them, so
    # they are oriented against each other pairwise instead.
    if len(images) == 1:
        return images[0].astype(np.float32)

    transforms = [FlipTransform()]
    shifts = [np.zeros(2, dtype=int)]
    for reference, moving in zip(images[:-1], images[1:]):
        scale = min(_registration_scale(reference), _registration_scale(moving))
        transform, shift, _ = estimate_flip_transform(
            reference, moving, channel_axis=channel_axis, scale=scale
        )
        # The shift was measured in the previous frame's orientation, so any axis
        # already mirrored by the running transform reverses its sign.
        shift = np.array(
            [-value if axis in transforms[-1] else value for axis, value in enumerate(shift)]
        )
        transforms.append(transforms[-1] + transform)
        shifts.append(shifts[-1] + shift)

    oriented = [transform(image) for transform, image in zip(transforms, images)]
    return mean_registered_image(oriented, np.array(shifts))


def align_to_atlas(
    images: list[np.ndarray],
    atlas: list[np.ndarray],
    n_queries: int = ATLAS_QUERIES,
) -> tuple["FlipTransform", float]:
    """
    Decide which way a self-consistent series faces, by comparison against an atlas.

    Evenly spaced frames are each matched against the atlas image closest to them in
    shape. The series is then flipped according to whichever way the majority of
    those matches point along the head-tail axis.

    Parameters:
        images (list[np.ndarray]): The series, already oriented to agree with itself.
        atlas (list[np.ndarray]): Manually oriented reference images, head left and
            vulva up.
        n_queries (int): Roughly how many frames to compare. (default: ATLAS_QUERIES)

    Returns:
        tuple[FlipTransform, float]: The transform bringing the series into the
            atlas' orientation, and the proportion of queries that agreed on the
            head-tail direction.
    """
    queries = images[:: max(len(images) // n_queries, 1)]

    tally: dict[FlipTransform, int] = {}
    for query in queries:
        reference = closest_shape_match(query, atlas)
        scale = min(_registration_scale(reference), _registration_scale(query))
        transform, _, _ = estimate_flip_transform(reference, query, scale=scale)
        tally[transform] = tally.get(transform, 0) + 1

    length_axis = images[0].ndim - 1
    flipped = {
        transform: count for transform, count in tally.items() if length_axis in transform
    }
    kept = {
        transform: count
        for transform, count in tally.items()
        if length_axis not in transform
    }
    majority = flipped if sum(flipped.values()) > sum(kept.values()) else kept
    confidence = sum(majority.values()) / sum(tally.values())
    return max(majority, key=majority.get), confidence


def predict_orientations(
    images: list[np.ndarray],
    atlas: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, str]], float]:
    """
    Predict the orientation of every image in one point's series.

    Parameters:
        images (list[np.ndarray]): The straightened images of one point, in time
            order, as single-channel arrays of shape ``(H, W)``.
        atlas (list[np.ndarray]): Manually oriented reference images, head left and
            vulva up.

    Returns:
        tuple[np.ndarray, list[dict[str, str]], float]: The registration shifts of
            shape ``(N, 2)``, the ``head`` / ``vulva`` labels of each image, and the
            proportion of atlas queries that agreed on the head-tail direction.
    """
    normalized = normalize_images_to_common_range(images)
    oriented, transforms, shifts = running_mean_orientation(normalized)
    consensus, confidence = align_to_atlas(oriented, atlas)

    # Flipping the whole series reverses coordinates along the mirrored axes, so
    # the shifts measured before the flip change sign with them.
    for axis in range(shifts.shape[-1]):
        if axis in consensus:
            shifts[:, axis] *= -1

    orientations = [
        (transform + consensus).orientation_labels() for transform in transforms
    ]
    return shifts, orientations, confidence
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: 28 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add straightened_movies.py tests/test_straightened_movies.py
git commit -m "feat: port orientation estimation onto the new geometry helpers

The mean-image path now asks registration_origins for a canvas centred on the
registration origin instead of padding to twice the largest frame and cropping
symmetrically afterwards. pairwise_image_transform survives only as the seed for
the running mean, so it becomes private, and the unused per-frame error return
is dropped."
```

---

### Task 6: Movie assembly

**Files:**
- Modify: `straightened_movies.py`
- Test: `tests/test_straightened_movies.py`

**Interfaces:**
- Consumes: `ALIGNMENTS`, `registration_origins`, `assemble_registered_stack` (Task 2)
- Produces:
  - `orient_images(images, orientations) -> list[np.ndarray]`
  - `build_movie(images, shifts, alignment="center") -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_straightened_movies.py`:

```python
# ---- Movie assembly ----


def test_orient_images_flips_by_label():
    image = np.array([[1, 2], [3, 4]])
    orientations = [
        {"head": "L", "vulva": "U"},
        {"head": "R", "vulva": "U"},
        {"head": "L", "vulva": "D"},
    ]

    oriented = sm.orient_images([image, image, image], orientations)

    assert np.array_equal(oriented[0], image)
    assert np.array_equal(oriented[1], np.array([[2, 1], [4, 3]]))
    assert np.array_equal(oriented[2], np.array([[3, 4], [1, 2]]))


def _growing_series(lengths, height=10):
    # Frames of growing length, each filled with a constant so the occupied
    # region of the canvas is easy to find.
    return [np.ones((height, length), dtype=np.uint16) for length in lengths]


def test_build_movie_left_keeps_the_head_end_stationary():
    images = _growing_series([10, 20, 30, 40])
    shifts = np.zeros((4, 2), dtype=int)

    movie = sm.build_movie(images, shifts, alignment="left")

    assert movie.shape == (4, 1, 10, 40)
    # Every frame starts at the same column, so growth is entirely rightwards.
    starts = [np.flatnonzero(frame[0].any(axis=0))[0] for frame in movie]
    assert starts == [0, 0, 0, 0]


def test_build_movie_center_spreads_growth_both_ways():
    images = _growing_series([10, 20, 30, 40])
    shifts = np.zeros((4, 2), dtype=int)

    movie = sm.build_movie(images, shifts, alignment="center")

    starts = [np.flatnonzero(frame[0].any(axis=0))[0] for frame in movie]
    ends = [np.flatnonzero(frame[0].any(axis=0))[-1] for frame in movie]
    # The head end moves left as the worm grows, and the tail end moves right.
    assert starts == sorted(starts, reverse=True)
    assert starts[0] > starts[-1]
    assert ends == sorted(ends)


def test_build_movie_preserves_registration_under_both_alignments():
    images = _growing_series([20, 20, 20, 20])
    shifts = np.array([[0, 0], [0, 4], [0, -3], [0, 7]])

    left = sm.build_movie(images, shifts, alignment="left")
    centered = sm.build_movie(images, shifts, alignment="center")

    def starts(movie):
        return [int(np.flatnonzero(frame[0].any(axis=0))[0]) for frame in movie]

    # Alignment shifts every frame by a constant; the spacing between frames,
    # which is what removes the jitter, is identical either way.
    assert np.array_equal(np.diff(starts(left)), np.diff(starts(centered)))
    assert np.array_equal(np.diff(starts(left)), np.diff(shifts[:, 1]))


def test_build_movie_promotes_single_channel_frames():
    movie = sm.build_movie(_growing_series([10, 12]), np.zeros((2, 2), dtype=int))

    assert movie.ndim == 4
    assert movie.shape[1] == 1


def test_build_movie_rejects_an_unknown_alignment():
    with pytest.raises(ValueError, match="alignment must be one of"):
        sm.build_movie(_growing_series([10]), np.zeros((1, 2), dtype=int), "diagonal")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v -k "orient_images or build_movie"`
Expected: FAIL — `AttributeError: module 'straightened_movies' has no attribute 'orient_images'`.

- [ ] **Step 3: Implement the movie section**

Append to `straightened_movies.py`:

```python
# ---------------------------------------------------------------------------
# Movie assembly
# ---------------------------------------------------------------------------


def orient_images(
    images: list[np.ndarray],
    orientations: list[dict[str, str]],
) -> list[np.ndarray]:
    """
    Flip each image so that the worm faces head left and vulva up.

    Parameters:
        images (list[np.ndarray]): Images of shape ``(..., H, W)``.
        orientations (list[dict[str, str]]): The ``head`` and ``vulva`` label of each
            image, as returned by ``predict_orientations``.

    Returns:
        list[np.ndarray]: The flipped images.
    """
    oriented = []
    for image, orientation in zip(images, orientations):
        if orientation["head"] == "R":
            image = np.flip(image, axis=-1)
        if orientation["vulva"] == "D":
            image = np.flip(image, axis=-2)
        oriented.append(image)
    return oriented


def build_movie(
    images: list[np.ndarray],
    shifts: np.ndarray,
    alignment: str = "center",
) -> np.ndarray:
    """
    Assemble registered frames into a movie.

    The frames are placed on a shared canvas by their registration shifts, which
    removes the jitter between consecutive timepoints, and anchored according to
    ``alignment``. Frames must already face head left and vulva up.

    Parameters:
        images (list[np.ndarray]): Frames of shape ``(C, H, W)`` or ``(H, W)``, in
            time order.
        shifts (np.ndarray): Registration shift of each frame, of shape ``(N, 2)``.
        alignment (str): ``"center"`` to let the worm grow in both directions, or
            ``"left"`` to anchor the head end so that it only grows rightwards.
            (default: "center")

    Returns:
        np.ndarray: The movie, of shape ``(T, C, H, W)``.

    Raises:
        ValueError: If ``alignment`` is not one of the known alignments.
    """
    if alignment not in ALIGNMENTS:
        raise ValueError(
            f"alignment must be one of {sorted(ALIGNMENTS)}, got {alignment!r}"
        )
    frames = [image[np.newaxis, ...] if image.ndim == 2 else image for image in images]
    shapes = np.array([frame.shape[-2:] for frame in frames])
    origins, canvas = registration_origins(
        shapes, shifts, anchors=ALIGNMENTS[alignment]
    )
    return assemble_registered_stack(frames, origins, canvas)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: 34 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add straightened_movies.py tests/test_straightened_movies.py
git commit -m "feat: make movie frame alignment selectable

Alignment reduces to an anchor vector: 'center' keeps frames centred as before,
'left' anchors the head end so a growing worm only extends rightwards. The
spacing between frames is the difference in their registration shifts under
either anchor, so the jitter correction is unaffected by the choice."
```

---

### Task 7: Block plumbing and end-to-end run

**Files:**
- Modify: `straightened_movies.py`
- Test: `tests/test_straightened_movies.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `DEFAULT_QC_VALUE = "worm"`
  - `load_filemap(path) -> pl.DataFrame`
  - `load_atlas(path, channel) -> list[np.ndarray]`
  - `read_straightened_image(path, channel) -> np.ndarray | None`
  - `parse_points(values) -> set[int] | None`
  - `resolve_movie_columns(requested, filemap) -> list[str]`
  - `predict_point_orientations(point, times, paths, atlas, channel) -> list[dict]`
  - `write_point_movies(point, rows, columns, movie_dir, alignment, max_frames) -> None`
  - `get_args() -> argparse.Namespace`
  - `main() -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_straightened_movies.py`:

```python
# ---- Block plumbing ----


def test_parse_points_accepts_numbers_and_ranges():
    assert sm.parse_points(["All"]) is None
    assert sm.parse_points(["3", "10-12", "42"]) == {3, 10, 11, 12, 42}


def test_parse_points_rejects_a_malformed_range():
    with pytest.raises(ValueError, match="Could not parse"):
        sm.parse_points(["1-2-3"])


def test_resolve_movie_columns_finds_straightened_columns():
    import polars as pl

    filemap = pl.DataFrame(
        {
            "Time": [0],
            "Point": [1],
            "analysis/ch1_raw_str": ["a.tiff"],
            "analysis/ch2_seg_str": ["b.tiff"],
            "ch2_seg_str_volume": [1.0],
        }
    )

    assert sm.resolve_movie_columns(["All"], filemap) == [
        "analysis/ch1_raw_str",
        "analysis/ch2_seg_str",
    ]
    assert sm.resolve_movie_columns(["analysis/ch1_raw_str"], filemap) == [
        "analysis/ch1_raw_str"
    ]


def test_resolve_movie_columns_rejects_a_missing_column():
    import polars as pl

    filemap = pl.DataFrame({"Time": [0], "Point": [1]})

    with pytest.raises(ValueError, match="not in the filemap"):
        sm.resolve_movie_columns(["analysis/nope_str"], filemap)


def test_read_straightened_image_returns_none_for_an_unreadable_file(tmp_path):
    broken = tmp_path / "broken.tiff"
    broken.write_bytes(b"not a tiff")

    assert sm.read_straightened_image(str(broken), 0) is None


@pytest.fixture
def synthetic_experiment(tmp_path):
    """A tiny two-point experiment: filemap, straightened images, and an atlas."""
    import polars as pl
    import tifffile

    experiment = tmp_path / "experiment"
    straightened = experiment / "analysis" / "ch1_raw_str"
    report = experiment / "analysis" / "report"
    atlas_dir = tmp_path / "atlas"
    for directory in (straightened, report, atlas_dir):
        directory.mkdir(parents=True)

    rows = []
    for point in (1, 2):
        for time, length in enumerate(range(30, 54, 4)):
            # Two channels; channel 1 carries the orientable structure. Point 2 is
            # imaged facing the other way.
            worm = _worm(length, flip=1 if point == 2 else None)
            image = np.stack([np.zeros_like(worm), worm])
            image = (image * 10_000).astype(np.uint16)
            path = straightened / f"Time{time:06d}_Point{point:06d}_str.tiff"
            tifffile.imwrite(str(path), image, photometric="minisblack")
            rows.append(
                {
                    "Time": time,
                    "Point": point,
                    "analysis/ch1_raw_str": str(path),
                    "ch1_seg_str_qc": "worm",
                }
            )

    filemap_path = report / "analysis_filemap.csv"
    pl.DataFrame(rows).write_csv(str(filemap_path))

    atlas_rows = []
    for length in (30, 40, 50, 60):
        worm = _worm(length)
        image = np.stack([np.zeros_like(worm), worm])
        image = (image * 10_000).astype(np.uint16)
        path = atlas_dir / f"atlas_{length}.tiff"
        tifffile.imwrite(str(path), image, photometric="minisblack")
        atlas_rows.append({"atlas_image": str(path)})

    atlas_csv = atlas_dir / "atlas.csv"
    pl.DataFrame(atlas_rows).write_csv(str(atlas_csv))

    return {
        "experiment": experiment,
        "filemap": filemap_path,
        "atlas": atlas_csv,
        "report": report,
    }


def _run_script(experiment, extra=()):
    import subprocess

    return subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "straightened_movies.py"),
            "--filemap", str(experiment["filemap"]),
            "--straightened_column", "analysis/ch1_raw_str",
            "--orientation_channel", "1",
            "--qc_column", "ch1_seg_str_qc",
            "--atlas", str(experiment["atlas"]),
            "--atlas_channel", "1",
            "--n_jobs", "1",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_end_to_end_writes_movies_and_a_cache(synthetic_experiment):
    import polars as pl

    result = _run_script(synthetic_experiment, ["--alignment", "left"])

    assert result.returncode == 0, result.stdout + result.stderr

    movies = synthetic_experiment["experiment"] / "movies" / "ch1_raw_str_movies"
    assert sorted(path.name for path in movies.iterdir()) == [
        "Point0001_movie.tiff",
        "Point0002_movie.tiff",
    ]

    cache = pl.read_csv(str(synthetic_experiment["report"] / "orientations.csv"))
    assert cache.columns == [
        "Time",
        "Point",
        "head",
        "vulva",
        "head_confidence",
        "shift_ax0",
        "shift_ax1",
    ]
    assert cache.height == 12
    # Point 2 was imaged facing the other way, so the two points get opposite labels.
    heads = {
        point: set(cache.filter(pl.col("Point") == point)["head"].to_list())
        for point in (1, 2)
    }
    assert heads[1] != heads[2]


def test_end_to_end_leaves_the_filemap_untouched(synthetic_experiment):
    before = synthetic_experiment["filemap"].read_text()

    _run_script(synthetic_experiment)

    assert synthetic_experiment["filemap"].read_text() == before


def test_end_to_end_reuses_the_cache_on_a_second_run(synthetic_experiment):
    import tifffile

    _run_script(synthetic_experiment, ["--alignment", "center"])
    cache_path = synthetic_experiment["report"] / "orientations.csv"
    cached = cache_path.read_text()

    movies = synthetic_experiment["experiment"] / "movies" / "ch1_raw_str_movies"
    centered = tifffile.imread(str(movies / "Point0001_movie.tiff"))

    result = _run_script(synthetic_experiment, ["--alignment", "left"])
    assert result.returncode == 0, result.stdout + result.stderr

    # The orientations are reused verbatim, but the movie is re-assembled with the
    # new alignment.
    assert cache_path.read_text() == cached
    assert "Reusing cached orientations" in result.stdout
    left = tifffile.imread(str(movies / "Point0001_movie.tiff"))
    assert left.shape[-1] < centered.shape[-1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v -k "parse_points or movie_columns or end_to_end or read_straightened"`
Expected: FAIL — `AttributeError: module 'straightened_movies' has no attribute 'parse_points'`.

- [ ] **Step 3: Add the remaining imports**

Add to the import block of `straightened_movies.py`:

```python
import argparse
import os
import pickle
import time
import traceback

import polars as pl
from joblib import delayed
from joblib import Parallel
from tifffile import imwrite
from towbintools.foundation.file_handling import read_filemap
from towbintools.foundation.file_handling import write_filemap
from towbintools.foundation.image_handling import read_tiff_file
```

- [ ] **Step 4: Implement the plumbing section**

Append to `straightened_movies.py`:

```python
# ---------------------------------------------------------------------------
# Pipeline block
# ---------------------------------------------------------------------------

# Quality-control label marking an image as a usable worm.
DEFAULT_QC_VALUE = "worm"
# Columns of the orientation cache, in order.
ORIENTATION_COLUMNS = [
    "Time",
    "Point",
    "head",
    "vulva",
    "head_confidence",
    "shift_ax0",
    "shift_ax1",
]


def load_filemap(path: str) -> pl.DataFrame:
    """
    Read a filemap from either a pickled DataFrame or a filemap file.

    The pipeline hands its workers a pickle; a manual run points at the experiment's
    own ``.csv`` or ``.parquet``.

    Parameters:
        path (str): Path to a ``.pkl``, ``.csv`` or ``.parquet`` file.

    Returns:
        pl.DataFrame: The filemap.
    """
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            return pl.DataFrame(pickle.load(handle))
    return read_filemap(path)


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


def read_straightened_image(path: str, channel: int) -> np.ndarray | None:
    """
    Read one channel of a straightened image, or ``None`` if it cannot be read.

    A small fraction of straightened images are written out malformed, and asking
    for a channel they do not have raises. Returning ``None`` lets the caller drop
    that frame instead of losing the whole point to it.

    Parameters:
        path (str): Path to the image.
        channel (int): Channel index within the image.

    Returns:
        np.ndarray | None: The channel as a 2D array, or ``None``.
    """
    try:
        image = read_tiff_file(path, channels_to_keep=[channel])
    except Exception as error:
        print(f"Could not read {path}: {error}")
        return None
    if image.ndim != 2:
        print(f"Unexpected shape {image.shape} for channel {channel} of {path}")
        return None
    return image


def load_atlas(path: str, channel: int) -> list[np.ndarray]:
    """
    Read the atlas images listed in an atlas CSV.

    Parameters:
        path (str): Path to a CSV with an ``atlas_image`` column.
        channel (int): Channel index within the atlas images, which must be the same
            kind of microscopy as the images being oriented.

    Returns:
        list[np.ndarray]: The atlas images, normalised against a shared range.

    Raises:
        ValueError: If no atlas image could be read.
    """
    listing = read_filemap(path)
    images = [
        image
        for image in (
            read_straightened_image(row, channel)
            for row in listing["atlas_image"].to_list()
        )
        if image is not None
    ]
    if not images:
        raise ValueError(f"No readable atlas images listed in {path}")
    return normalize_images_to_common_range(images)


def parse_points(values: list[str]) -> set[int] | None:
    """
    Parse a point selection into the set of points to process.

    Parameters:
        values (list[str]): ``["All"]``, or point numbers and inclusive ranges such
            as ``["3", "10-20", "42"]``.

    Returns:
        set[int] | None: The selected points, or ``None`` for all of them.

    Raises:
        ValueError: If an entry is neither a number nor a range.
    """
    if len(values) == 1 and values[0] == "All":
        return None

    points = set()
    for value in values:
        try:
            if "-" in value:
                first, last = value.split("-")
                points.update(range(int(first), int(last) + 1))
            else:
                points.add(int(value))
        except ValueError:
            raise ValueError(f"Could not parse {value!r} as a point number or range")
    return points


def resolve_movie_columns(requested: list[str], filemap: pl.DataFrame) -> list[str]:
    """
    Work out which filemap columns to build movies from.

    Parameters:
        requested (list[str]): ``["All"]`` for every column of straightened images,
            or an explicit list of column names.
        filemap (pl.DataFrame): The experiment filemap.

    Returns:
        list[str]: The columns to process.

    Raises:
        ValueError: If a requested column is not in the filemap, or if ``All``
            matched nothing.
    """
    if len(requested) == 1 and requested[0] == "All":
        columns = [column for column in filemap.columns if column.endswith("_str")]
        if not columns:
            raise ValueError("No column ending in '_str' found in the filemap")
        return columns

    missing = [column for column in requested if column not in filemap.columns]
    if missing:
        raise ValueError(f"Requested columns not in the filemap: {missing}")
    return requested


def predict_point_orientations(
    point: int,
    times: list[int],
    paths: list[str],
    atlas: list[np.ndarray],
    channel: int,
) -> list[dict]:
    """
    Predict orientations and registration shifts for one point.

    Parameters:
        point (int): The point being processed.
        times (list[int]): Timepoints, in order, matching ``paths``.
        paths (list[str]): Straightened image paths, in time order.
        atlas (list[np.ndarray]): Manually oriented reference images.
        channel (int): Channel index used to predict orientation.

    Returns:
        list[dict]: One record per readable image, with the orientation-cache
            columns. Empty if the point could not be processed.
    """
    try:
        images = [read_straightened_image(path, channel) for path in paths]
        readable = [image is not None for image in images]
        if not all(readable):
            print(
                f"Point {point}: skipped {readable.count(False)}/{len(readable)} "
                "images that could not be read"
            )
        times = [t for t, keep in zip(times, readable) if keep]
        images = [image for image in images if image is not None]
        if not images:
            raise ValueError(f"No readable images for point {point}")

        shifts, orientations, confidence = predict_orientations(images, atlas)

        return [
            {
                "Time": t,
                "Point": point,
                "head": orientation["head"],
                "vulva": orientation["vulva"],
                "head_confidence": confidence,
                "shift_ax0": int(shift[0]),
                "shift_ax1": int(shift[1]),
            }
            for t, orientation, shift in zip(times, orientations, shifts)
        ]
    except Exception:
        print(f"Error predicting orientations for point {point}")
        print(traceback.format_exc())
        return []


def write_point_movies(
    point: int,
    rows: pl.DataFrame,
    columns: list[str],
    movie_dir: str,
    alignment: str,
    max_frames: int | None,
) -> None:
    """
    Assemble and write one point's movies, one per column of straightened images.

    Parameters:
        point (int): The point being processed.
        rows (pl.DataFrame): That point's rows, in time order, carrying the image
            paths and the ``head`` / ``vulva`` / ``shift_ax0`` / ``shift_ax1``
            columns.
        columns (list[str]): Filemap columns to build movies from.
        movie_dir (str): Directory the per-column movie directories are created in.
        alignment (str): ``"center"`` or ``"left"``.
        max_frames (int | None): Cap on the number of frames, or ``None`` for all.
    """
    if max_frames is not None:
        rows = rows.head(max_frames)

    orientations = rows.select(["head", "vulva"]).to_dicts()
    shifts = rows.select(["shift_ax0", "shift_ax1"]).to_numpy().astype(int)

    for column in columns:
        output_dir = os.path.join(movie_dir, f"{os.path.basename(column)}_movies")
        os.makedirs(output_dir, exist_ok=True)
        try:
            images = [read_tiff_file(path) for path in rows[column].to_list()]
            movie = build_movie(
                orient_images(images, orientations), shifts, alignment=alignment
            )
            imwrite(
                os.path.join(output_dir, f"Point{point:04}_movie.tiff"),
                movie,
                compression="zlib",
                imagej=True,
                metadata={"axes": "TCYX"},
            )
        except Exception:
            print(f"Error building the {column} movie for point {point}")
            print(traceback.format_exc())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # The pipeline's custom-block contract. --block_config and --output are accepted
    # so that the generated command line is satisfied; this block writes neither a
    # report nor a directory the filemap knows about.
    parser.add_argument("-f", "--filemap", required=True)
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-b", "--block_config", default=None)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-j", "--n_jobs", type=int, default=None)

    parser.add_argument("--straightened_column", required=True)
    parser.add_argument("--orientation_channel", type=int, default=0)
    parser.add_argument("--atlas", default=None)
    parser.add_argument("--atlas_channel", type=int, default=0)
    parser.add_argument("--qc_column", default=None)
    parser.add_argument("--qc_value", default=DEFAULT_QC_VALUE)
    parser.add_argument("--movie_columns", nargs="+", default=["All"])
    parser.add_argument("--movie_dir", default=None)
    parser.add_argument("--orientations", default=None)
    parser.add_argument("--recompute_orientations", action="store_true")
    parser.add_argument("--alignment", choices=sorted(ALIGNMENTS), default="center")
    parser.add_argument("--points", nargs="+", default=["All"])
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


def _resolve_directories(args: argparse.Namespace, config: dict) -> tuple[str, str]:
    # The pipeline config carries the experiment layout. Without it, fall back to
    # the filemap's own location: <experiment>/<analysis>/report/filemap.csv.
    report_dir = config.get("report_subdir")
    experiment_dir = config.get("experiment_dir")
    if report_dir is None:
        report_dir = os.path.dirname(os.path.abspath(args.filemap))
    if experiment_dir is None:
        experiment_dir = os.path.dirname(os.path.dirname(report_dir))

    movie_dir = args.movie_dir or os.path.join(experiment_dir, "movies")
    orientations = args.orientations or os.path.join(report_dir, "orientations.csv")
    return movie_dir, orientations


def main() -> None:
    args = get_args()
    config = load_config(args.config)
    n_jobs = args.n_jobs or config.get("n_jobs") or int(
        os.environ.get("SLURM_CPUS_PER_TASK", 1)
    )
    movie_dir, orientations_path = _resolve_directories(args, config)

    filemap = load_filemap(args.filemap)
    if args.straightened_column not in filemap.columns:
        raise ValueError(
            f"Column {args.straightened_column!r} is not in the filemap. "
            f"Available columns: {filemap.columns}"
        )

    rows = filemap.select(
        [
            column
            for column in dict.fromkeys(
                ["Time", "Point", args.straightened_column]
                + resolve_movie_columns(args.movie_columns, filemap)
                + ([args.qc_column] if args.qc_column else [])
            )
        ]
    )
    if args.qc_column:
        rows = rows.filter(pl.col(args.qc_column) == args.qc_value)
    selected = parse_points(args.points)
    if selected is not None:
        rows = rows.filter(pl.col("Point").is_in(sorted(selected)))
    rows = rows.sort(["Point", "Time"])
    if rows.height == 0:
        raise ValueError("No images left to process after filtering")

    movie_columns = resolve_movie_columns(args.movie_columns, filemap)
    points = rows["Point"].unique().sort().to_list()
    print(f"{time.asctime()} - Processing {len(points)} points: {points}")

    cached = None
    if not args.recompute_orientations and os.path.exists(orientations_path):
        cached = read_filemap(orientations_path).select(ORIENTATION_COLUMNS)
        covered = set(cached["Point"].unique().to_list())
        print(f"Reusing cached orientations from {orientations_path}")
    else:
        covered = set()

    missing = [point for point in points if point not in covered]
    if missing:
        if args.atlas is None:
            raise ValueError(
                f"--atlas is required: {len(missing)} points are not in the cache"
            )
        atlas = load_atlas(args.atlas, args.atlas_channel)
        started = time.time()
        print(f"{time.asctime()} - Predicting orientations for {len(missing)} points")
        records = Parallel(n_jobs=n_jobs)(
            delayed(predict_point_orientations)(
                point,
                rows.filter(pl.col("Point") == point)["Time"].to_list(),
                rows.filter(pl.col("Point") == point)[args.straightened_column].to_list(),
                atlas,
                args.orientation_channel,
            )
            for point in missing
        )
        predicted = pl.DataFrame(
            [record for point_records in records for record in point_records],
            schema=ORIENTATION_COLUMNS,
        )
        cached = (
            predicted
            if cached is None
            else pl.concat([cached, predicted]).unique(
                subset=["Time", "Point"], keep="last"
            )
        )
        cached = cached.sort(["Point", "Time"])
        os.makedirs(os.path.dirname(orientations_path), exist_ok=True)
        write_filemap(cached, orientations_path)
        print(
            f"{time.asctime()} - Orientations written to {orientations_path} "
            f"({round((time.time() - started) / 60)}min)"
        )

    movie_rows = rows.join(cached, on=["Time", "Point"], how="inner").sort(
        ["Point", "Time"]
    )
    started = time.time()
    print(f"{time.asctime()} - Building {args.alignment}-aligned movies in {movie_dir}")
    Parallel(n_jobs=n_jobs)(
        delayed(write_point_movies)(
            point,
            movie_rows.filter(pl.col("Point") == point),
            movie_columns,
            movie_dir,
            args.alignment,
            args.max_frames,
        )
        for point in points
    )
    print(
        f"{time.asctime()} - Finished ({round((time.time() - started) / 60)}min)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: 43 passed.

- [ ] **Step 6: Run the formatters**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m black straightened_movies.py tests/ && ~/.local/bin/micromamba run -n towbintools python -m isort straightened_movies.py tests/`
Then re-run the tests to confirm nothing broke.

- [ ] **Step 7: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add straightened_movies.py tests/test_straightened_movies.py
git commit -m "feat: run orientation and movie generation as one block

Combines the two scripts behind one entry point that takes either the pipeline's
pickled filemap or a filemap file, so the same code serves a chained block and a
manual run. The orientation table is written as a re-run cache only: nothing here
is registered in the experiment filemap."
```

---

### Task 8: Settle the translation scoring on real data

**Files:**
- Modify: `straightened_movies.py` (only if the measurement says so)
- Create: `docs/translation-scoring-check.md`

**Interfaces:**
- Consumes: `estimate_translation`, `predict_orientations` (Tasks 3, 5)
- Produces: a decision, recorded in `docs/translation-scoring-check.md`

**Context:** `legacy/ori_align.py:232` called `np.roll(mov_padded, shift)` with no
`axis`, which flattens the array and rolls it by `dy + dx`. Task 3 ships the fixed
version. This task checks the fix against real data before it stands.

- [ ] **Step 1: Write the comparison script**

Create `/tmp/claude-1000060/-home-spsalmon-straightened-movies/cc5ea152-e624-4f6d-8440-1686c901a590/scratchpad/compare_scoring.py`:

```python
"""Compare flattened vs per-axis rolling in estimate_translation, on real data."""

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "/home/spsalmon/straightened_movies")
import straightened_movies as sm

EXPERIMENT = (
    "/mnt/towbin.data/shared/kstojanovski/"
    "20240202_Orca_10x_yap-1del_col-10-tir_wBT160-186-310-337-380-393_25C_"
    "20240202_171239_051"
)
FILEMAP = os.path.join(EXPERIMENT, "analysis_sacha/report/analysis_filemap.csv")
COLUMN = "analysis_sacha/ch1_ch2_raw_str"
QC = "ch2_seg_str_worm_type"
ATLAS = "/mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv"

original = sm.estimate_translation


def flattened_roll(reference, moving):
    # The historical behaviour: np.roll with no axis argument.
    saved = np.roll

    def rolled(array, shift, axis=None):
        return saved(array, shift) if axis == (-2, -1) else saved(array, shift, axis)

    np.roll = rolled
    try:
        return original(reference, moving)
    finally:
        np.roll = saved


def confidences(points):
    atlas = sm.load_atlas(ATLAS, 1)
    filemap = sm.read_filemap(FILEMAP).filter(pl.col(QC) == "worm").sort(["Point", "Time"])
    results = {}
    for point in points:
        paths = filemap.filter(pl.col("Point") == point)[COLUMN].to_list()
        images = [sm.read_straightened_image(path, 1) for path in paths]
        images = [image for image in images if image is not None]
        _, _, confidence = sm.predict_orientations(images, atlas)
        results[point] = confidence
    return results


POINTS = [1, 2, 3, 4, 5]

print("per-axis roll (the fix):")
fixed = confidences(POINTS)
print(fixed, "mean", round(np.mean(list(fixed.values())), 4))

sm.estimate_translation = flattened_roll
print("flattened roll (historical):")
historical = confidences(POINTS)
print(historical, "mean", round(np.mean(list(historical.values())), 4))
```

- [ ] **Step 2: Run the comparison**

Run: `~/.local/bin/micromamba run -n towbintools python /tmp/claude-1000060/-home-spsalmon-straightened-movies/cc5ea152-e624-4f6d-8440-1686c901a590/scratchpad/compare_scoring.py`

This reads a real experiment and takes several minutes. If the experiment is
unreachable, record that in Step 3 and keep the fix (it is what the surrounding
code intends), then stop.

- [ ] **Step 3: Record the result**

Create `docs/translation-scoring-check.md` with the measured `head_confidence` per
point for both variants, the means, and the conclusion. State which variant ships
and why.

- [ ] **Step 4: Apply the decision**

If the historical variant scores meaningfully better, change `estimate_translation`
in `straightened_movies.py` to roll without an axis and add a comment saying the
per-axis roll was measured and rejected, citing the document. Otherwise leave the
code as it is. Re-run the full test suite either way:

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: 43 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add docs/translation-scoring-check.md straightened_movies.py
git commit -m "docs: record the translation-scoring measurement on real data

The original rolled the moving image without an axis argument, scoring candidate
flips on a misaligned image. Measured both variants on five points of the
20240202 experiment before settling on one."
```

---

### Task 9: Runner, config and documentation

**Files:**
- Create: `run_straightened_movies.slurm`, `configs/straightened_movies.yaml`, `README.md`

**Interfaces:**
- Consumes: the CLI from Task 7
- Produces: nothing other tasks depend on

- [ ] **Step 1: Write `run_straightened_movies.slurm`**

```bash
#!/bin/bash

#SBATCH --job-name="str_movies"
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Orientation prediction dominates the runtime and parallelises over points, so
# there is nothing to gain from more CPUs than there are points. Movie assembly is
# I/O bound: every straightened image is read back and written out as one stack.
# Re-runs over an experiment that already has an orientations.csv skip prediction
# entirely, which makes trying a different --alignment cheap.

echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

EXPERIMENT="/mnt/towbin.data/shared/USER/EXPERIMENT"

~/.local/bin/micromamba run -n towbintools python3 \
    "$(dirname "$0")/straightened_movies.py" \
    --filemap "${EXPERIMENT}/analysis/report/analysis_filemap.csv" \
    --straightened_column "analysis/ch1_ch2_raw_str" \
    --orientation_channel 1 \
    --qc_column "ch2_seg_str_qc" \
    --atlas "/mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv" \
    --atlas_channel 1 \
    --movie_columns All \
    --alignment left \
    --points All \
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
```

- [ ] **Step 2: Write `configs/straightened_movies.yaml`**

```yaml
# Example: straightened movies as a pipeline custom block. The block writes movies
# and an orientation cache; it adds nothing to the experiment filemap, which is what
# custom_script_return_type: null means. That return type needs the pipeline patch
# from this rewrite (see docs/superpowers/specs/).
experiment_dir: "/mnt/towbin.data/shared/USER/EXPERIMENT"
analysis_dir_name: "analysis"
raw_dir_name: "raw"
report_format: "csv"
pixelsize: [0.65]
get_experiment_time: True

sbatch_memory: 64G
sbatch_time: 0-12:00:00
sbatch_cpus: 32

building_blocks:
  - "custom"

custom_script_path: ["/home/spsalmon/straightened_movies/straightened_movies.py"]
custom_script_name: ["straightened_movies"]
custom_script_return_type: [null]
custom_script_parameters:
  - [
      "--straightened_column analysis/ch1_ch2_raw_str",
      "--orientation_channel 1",
      "--qc_column ch2_seg_str_qc",
      "--atlas /mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv",
      "--atlas_channel 1",
      "--movie_columns All",
      "--alignment left",
    ]
```

- [ ] **Step 3: Write `README.md`**

Cover, in this order: what the block does; the requirements (straightened images,
a QC column, an atlas); how to run it standalone and as a custom block; every CLI
option with the same wording as `--help`; the `center` versus `left` alignment
difference and the fact that the choice does not affect the registration; the
orientation cache and when it is reused; the output layout
(`<movie_dir>/<column>_movies/Point####_movie.tiff`); how to pick an atlas (it must
match the kind of microscopy, and the channel is the channel *within the image
file*, not the experimental channel); and a pointer to `legacy/` and to
`docs/superpowers/specs/`.

Carry over from `legacy/README.txt` the notes on `head_confidence` and on the
unreliability of the `vulva` label: worms are close to rotationally symmetric, the
vulva may not be visible at all in some channels, and a worm caught mid-rotation
gives a smooth U-to-D transition, so movies may show random or gradual up-down
flips.

- [ ] **Step 4: Check the script's help renders**

Run: `cd /home/spsalmon/straightened_movies && ~/.local/bin/micromamba run -n towbintools python straightened_movies.py --help`
Expected: the full option list, no traceback.

- [ ] **Step 5: Commit**

```bash
cd /home/spsalmon/straightened_movies
git add run_straightened_movies.slurm configs/straightened_movies.yaml README.md
git commit -m "docs: document the block, its options and the two alignments"
```

---

### Task 10: A pipeline block that produces no output

**Files:**
- Modify: `~/towbintools_pipeline/towbintools_pipeline/building_blocks.py:233-356` (`BuildingBlock.run`), `:652-680` (`CustomBuildingBlock.get_output_name`)
- Modify: `~/towbintools_pipeline/towbintools_pipeline/utils.py:668-675` (`create_linker_command`)
- Test: `~/towbintools_pipeline/tests/test_local_pipeline.py`

**Interfaces:**
- Consumes: nothing from the tasks above
- Produces: `custom_script_return_type: null` runs the script and chains to the next block without touching the filemap

**Context:** `BuildingBlock.run` branches only on `"subdir"` and `"csv"`. Any other
return type falls off the end: the block never runs and no linker command is
emitted, so the chain stops. `CustomBuildingBlock.get_output_name` raises
`UnboundLocalError` for the same reason, and its `if` / `if subdir is not None` /
`elif` sequence is also wrong for a `csv` block in a subdir experiment.

- [ ] **Step 1: Branch**

```bash
cd /home/spsalmon/towbintools_pipeline
git checkout -b feat/no-output-building-block
```

- [ ] **Step 2: Write the failing test**

Append to `~/towbintools_pipeline/tests/test_local_pipeline.py`:

```python
def test_custom_block_with_no_return_type_runs_and_chains(tmp_path):
    # A custom block that produces nothing the filemap knows about still has to run
    # and still has to hand over to the next block.
    script = tmp_path / "sentinel.py"
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-f', '--filemap')\n"
        "parser.add_argument('-c', '--config')\n"
        "parser.add_argument('-b', '--block_config')\n"
        "parser.add_argument('-o', '--output')\n"
        "parser.add_argument('--sentinel')\n"
        "args = parser.parse_args()\n"
        "open(args.sentinel, 'w').write('ran')\n"
    )
    sentinel = tmp_path / "sentinel.txt"

    config_path = _build_experiment(
        tmp_path,
        extra_config={
            "building_blocks": ["custom", "segmentation"],
            "custom_script_path": [str(script)],
            "custom_script_name": ["sentinel"],
            "custom_script_return_type": [None],
            "custom_script_parameters": [[f"--sentinel {sentinel}"]],
        },
    )

    result = _run_pipeline(config_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text() == "ran"
    # The block after it still ran, so the chain was not broken.
    masks = tmp_path / "exp" / "analysis" / "ch1_seg"
    assert masks.is_dir() and any(masks.iterdir())
    # And nothing was added to the filemap for it.
    with open(tmp_path / "exp" / "analysis" / "report" / "analysis_filemap.csv") as f:
        header = f.readline()
    assert "sentinel" not in header
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd /home/spsalmon/towbintools_pipeline && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/test_local_pipeline.py::test_custom_block_with_no_return_type_runs_and_chains -v`
Expected: FAIL — the sentinel file is never written, because the block does not run.

- [ ] **Step 4: Make `create_linker_command` tolerate no result**

In `towbintools_pipeline/utils.py`, replace `create_linker_command` with:

```python
def create_linker_command(
    python_command,
    temp_dir,
    result=None,
):
    # A block with no output has no result to record, so --result is left off
    # rather than passed as the string "None".
    linker_command = (
        f"{python_command} -m towbintools_pipeline.block_linker --temp_dir {temp_dir}"
    )
    if result is not None:
        linker_command += f" --result {result}"
    return linker_command
```

- [ ] **Step 5: Add the no-output branch to `BuildingBlock.run`**

In `towbintools_pipeline/building_blocks.py`, after the `elif self.return_type == "csv":` branch ends with `return output_file`, add:

```python
        else:
            # A block that produces nothing the filemap knows about: run it, then
            # chain to the next block with no result to record. The inputs are
            # pickled under the same name the csv branch uses, because
            # CustomBuildingBlock.create_command passes that pickle as --filemap.
            input_files, _ = self.get_input_and_output_files(
                config, experiment_filemap, config["analysis_subdir"]
            )

            input_pickle_path, pickled_block_config, pickled_config = pickle_objects(
                temp_dir,
                {"path": "input_files", "obj": input_files},
                {"path": "block_config", "obj": block_config},
                {"path": "config", "obj": config},
            )

            command = self.create_command(
                python_command,
                input_pickle_path,
                None,
                pickled_block_config,
                pickled_config,
                config,
                pickled_filemap_path=pickled_filemap_path,
            )

            run_command(
                command,
                self.name,
                config,
                requires_gpu=self.requires_gpu,
                run_linker=True,
                linker_command=create_linker_command(python_command, temp_dir),
            )

            return None
```

Note that `CustomBuildingBlock.get_input_and_output_files` returns
`(experiment_filemap, None)`, so for a custom block `input_files` *is* the filemap.
That is how `--filemap` is populated for the existing `csv` custom blocks, and the
branch above keeps that contract.

- [ ] **Step 6: Fix `CustomBuildingBlock.get_output_name`**

Replace the method body with:

```python
    def get_output_name(self, config, subdir):
        custom_script_name = self.block_config["custom_script_name"]

        if self.return_type == "subdir":
            output = os.path.join(config["analysis_subdir"], custom_script_name)
            if subdir is not None:
                output = os.path.join(output, subdir)
            return output

        if self.return_type == "csv":
            name = custom_script_name if subdir is None else f"{subdir}_{custom_script_name}"
            return os.path.join(
                config["report_subdir"], f"{name}.{config['report_format']}"
            )

        # A block with no return type has no output path.
        return None
```

- [ ] **Step 7: Check the generated command by hand**

`CustomBuildingBlock.create_command` interpolates `--output {output_pickle_path}`,
which is `None` for this return type and so renders as the string `"None"`. The
worker must therefore accept `--output None` and ignore it, which
`straightened_movies.py` does. Confirm the rendered command is well formed:

Run: `cd /home/spsalmon/towbintools_pipeline && ~/.local/bin/micromamba run -n towbintools python -c "
from towbintools_pipeline.building_blocks import CustomBuildingBlock
block = CustomBuildingBlock({
    'custom_script_path': '/tmp/x.py', 'custom_script_name': 'x',
    'custom_script_return_type': None, 'custom_script_parameters': ['--flag 1'],
    'rerun_custom': False,
})
print(block.create_command('python', 'in.pkl', None, 'b.pkl', 'c.pkl', {}))
print(block.get_output_name({'analysis_subdir': 'a', 'report_subdir': 'r', 'report_format': 'csv'}, None))
"`
Expected: a single-line command containing `--filemap in.pkl`, `--output None` and
`--flag 1`, then `None` for the output name — no traceback.

- [ ] **Step 8: Run the test to verify it passes**

Run: `cd /home/spsalmon/towbintools_pipeline && ~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v`
Expected: all tests pass, including the new one.

- [ ] **Step 9: Document the return type**

In `book/building_blocks/`, find the notebook covering the custom block
(`grep -rl "custom_script_return_type" book/`) and add `null` to the documented
values for `custom_script_return_type`, saying that the block runs and chains but
contributes nothing to the filemap, and that it is the right choice for a block
whose output is not per-timepoint — a movie per point, for instance.

- [ ] **Step 10: Commit**

```bash
cd /home/spsalmon/towbintools_pipeline
git add -A
git commit -m "feat: support a building block that produces no filemap output

run() branched only on 'subdir' and 'csv', so any other return type fell through:
the block never ran and no linker command was emitted, stopping the chain. Blocks
whose output is not per-timepoint, such as one movie per point, have nothing to
register in the filemap and need to say so.

Also fixes CustomBuildingBlock.get_output_name, whose if / if / elif sequence
raised UnboundLocalError for a csv block in a subdir experiment."
```

- [ ] **Step 11: Report the branch**

Tell the user the branch name and that it is not merged, so they can review it
against `dev` before merging.

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: output contract →
Tasks 7 and 10; frame placement and the `center=True` mode → Task 2; alignment
independence → Tasks 2 and 6; library functions used → Tasks 3 and 7; script
structure sections 1-4 → Tasks 2-7; command line → Task 7; orientation cache →
Task 7; error handling → Tasks 5 and 7; pipeline patch → Task 10; tests → spread
across every task; repository layout → Tasks 1 and 9; style → the global
constraints. The translation-scoring bug is not in the spec because it was found
while writing this plan; Task 8 covers it.

**Placeholders.** None: every code step carries the code, and the two prose-only
deliverables (`README.md` in Task 9, `docs/translation-scoring-check.md` in Task 8)
enumerate exactly what must be in them.

**Type consistency.** `registration_origins` returns
`(origins, canvas)` and every caller unpacks two values. `assemble_registered_stack`
takes `(images, origins, canvas_shape, fill=0)` throughout. `FlipTransform` is the
name everywhere, with `orientation_labels()` used in Tasks 4, 5 and the tests.
`predict_orientations` returns `(shifts, orientations, confidence)` in that order in
Task 5 and is consumed that way in Task 7. `ORIENTATION_COLUMNS` matches the cache
columns asserted in the Task 7 test and the record keys built by
`predict_point_orientations`.
