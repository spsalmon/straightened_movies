# Straightened movies: rewrite on towbintools / towbintools_pipeline

Date: 2026-09-14

## Problem

`ori_align.py` (~820 lines) and `movie_generation.py` (~280 lines) predict worm
orientations from straightened images and assemble them into per-point movies.
They predate `towbintools` and `towbintools_pipeline` and therefore carry their
own copies of image padding, cropping, shift normalisation, filemap I/O and
argument handling. Four helpers (`pad_to_shape`, `rectangular_crop`,
`normalise_shifts`, `create_movie`) exist in both files in diverging versions.
Configuration is a block of module-level constants that must be edited in place,
which makes the scripts untestable and unusable from the pipeline.

Movies are also assembled with the frames centred on the canvas, so a growing
worm expands in both directions. Growth should be expressible as left-to-right
only, without discarding the frame-to-frame registration that removes jitter.

## Goals

1. One script, `straightened_movies.py`, doing orientation prediction and movie
   assembly, runnable both standalone and as a `towbintools_pipeline` custom
   building block.
2. Use `towbintools` / `towbintools_pipeline` functions wherever one fits.
   Generic pieces that do not yet exist are written in `towbintools` house style,
   grouped in one section, ready to be lifted upstream verbatim.
3. Selectable frame alignment along the head-tail axis: `center` (current
   behaviour) or `left` (head end anchored, growth to the right), with
   inter-frame registration preserved in both.
4. A pytest suite in the style of `towbintools_pipeline/tests/`: synthetic data
   generated at runtime, no bundled fixtures.

## Non-goals

- Changing the orientation algorithm. Flip estimation, SSIM dissimilarity,
  running-mean bootstrapping, atlas consensus and the error terms are
  behaviour-preserving.
- Registering movies, or anything else this block produces, in the experiment
  filemap.
- Building an atlas. Atlases stay pre-made CSVs, as today.

## Output contract

The block produces **no filemap-visible output**: `custom_script_return_type:
null`. Movies are written to `--movie_dir`, and the orientation table is written
to `--orientations` purely as a cache. Neither is joined into the filemap.

This contract does not exist in the pipeline yet and is added as part of this
work; see "Pipeline patch" below.

## Core design: frame placement

The current assembly is: normalise shifts, pad every frame into a canvas of
`2 x max_shape`, `np.roll` by the shift, then crop the black borders back off
using a parallel stack of boolean masks. The `left` variant threads a
`flush_axes` argument through all three steps.

This is replaced by computing each frame's origin in a shared canvas directly.
For a per-axis anchor `a` (0.0 = flush low-index edge, 0.5 = centred), a frame
padded into a canvas of size `T` starts at `a * (T - shape_i)` and is then
translated by its registration shift `s_i`:

```
origin_i = s_i + a * T - a * shape_i
```

`a * T` is constant across frames, so it cancels once origins are rebased:

```
raw     = shifts - rint(anchors * shapes)          # (N, 2)
origins = raw - raw.min(axis=0)                    # canvas starts at 0
canvas  = (origins + shapes).max(axis=0)           # tight union bounding box
```

Consequences:

- The canvas *is* the bounding box of the union of placed frames, so the
  mask-stack cropping step disappears.
- Nothing is rolled, so wrap-around is structurally impossible.
- `center` and `left` differ only in the anchor vector. Relative spacing between
  frames is `s_i - s_j` under every anchor, so registration is preserved exactly;
  only the common offset moves.

`ALIGNMENTS = {"center": (0.5, 0.5), "left": (0.5, 0.0)}` — the dorsoventral axis
(`-2`) stays centred in both, the head-tail axis (`-1`) is what changes. Under
`left`, the frame sitting furthest left defines `x = 0` and every other frame is
offset right by its registered amount, which is the requested "anchored but not
pinned to absolute zero" behaviour.

### Alignment is independent of orientation prediction

The orientation stage produces shifts that do not depend on the alignment: they
are measured between consecutive frames (and against a running mean) before any
canvas exists, and `registration_origins` rebases them, so adding a constant to
every shift changes nothing. A cache computed under `center` is therefore valid
under `left`, and vice versa. A test asserts this directly, by running
`build_movie` on one set of shifts under both alignments and checking that
inter-frame spacing is identical.

This rests on the shifts being jitter rather than drift, which was checked
against a real run (`legacy/example_output.csv`, 4688 rows): `shift_ax1` has
correlation 0.12 with `Time`, standard deviation 5.8 px and range [-25, 22],
with no growth trend. Had registration been anchoring on the head, `shift_ax1`
would instead track worm length. With `s_i` around zero:

- `center` gives `origin_i = s_i - W_i / 2`, frames centred, growth in both
  directions — the behaviour seen today.
- `left` gives `origin_i = s_i`, frames flush left up to the jitter that keeps
  them registered — the head anchored, growth to the right.

### The `center=True` mode

`running_mean_orientation` averages an assembled stack and feeds the result back
into `phase_cross_correlation`, which measures shifts relative to image
*centres*. A tight union bounding box is not generally centred on the
registration origin, so the mean-image path needs the canvas grown symmetrically
about it instead:

```
half    = maximum(-raw.min(axis=0), (raw + shapes).max(axis=0))
origins = raw + half
canvas  = 2 * half
```

This replaces the old `rectangular_crop(..., symmetric=True)` trick and keeps the
geometry logic in a single function.

## Library functions used

| Replaced | By |
|---|---|
| `tifffile.imread(f, key=n)` | `towbintools.foundation.image_handling.read_tiff_file(path, channels_to_keep=[n])` |
| `pad_to_shape` (centred, inside `estimate_translation`) | `image_handling.pad_to_dim_equally` |
| `pd.read_csv` / `to_csv` on filemaps | `foundation.file_handling.read_filemap` / `write_filemap` (polars) |
| pickle loading in the worker | `towbintools_pipeline.utils.load_pickles` |
| `pad_to_shape` + `np.roll` + `rectangular_crop` + `normalise_shifts`, duplicated across both scripts | `registration_origins` + `assemble_registered_stack` |

`image_handling.align_images_orientation_ssim` is close to the flip search but
returns only the flipped image, not the shift and error the consensus step needs,
so the flip machinery stays local.

`towbintools_pipeline.utils.basic_get_args` defines a fixed parser that cannot be
extended with the block's own options, so the script defines its own parser
mirroring the same flags.

## Script structure

`straightened_movies.py`, in four commented sections.

### 1. Generic helpers — candidates for towbintools

Self-contained, fully annotated, towbintools docstring style; no dependency on
anything below them.

```python
registration_origins(shapes, shifts, anchors=(0.5, 0.5), center=False)
    -> tuple[np.ndarray, tuple[int, int]]
assemble_registered_stack(images, origins, canvas_shape, fill=0) -> np.ndarray
crop_to_mask(image, mask) -> np.ndarray
normalize_images_to_common_range(images, out_range="float32") -> list[np.ndarray]
structural_dissimilarity(image, other, channel_axis=None) -> float
estimate_translation(reference, moving) -> tuple[np.ndarray, float]
class FlipTransform
estimate_flip_transform(reference, moving, channel_axis=None, scale=None)
    -> tuple[FlipTransform, np.ndarray, float]
```

`normalize_images_to_common_range` rescales a whole series per channel against a
range shared across the series, which is what the orientation stage needs and
what `image_handling.normalize_image` (per-image) does not provide.

`FlipTransform` keeps the existing semantics: a frozenset of mirrored axes,
composition through `__add__` (symmetric difference), self-inverse, and
`orientation_labels()` mapping to `head` L/R and `vulva` U/D against the
head-left / vulva-up convention.

### 2. Orientation estimation

```python
_auto_scale(image, max_size=40_000) -> float
closest_shape_match(image, references) -> np.ndarray
_bootstrap_mean_image(images, channel_axis=None) -> np.ndarray
running_mean_orientation(images, window=5, channel_axis=None)
    -> tuple[list[np.ndarray], list[FlipTransform], np.ndarray]
align_to_atlas(images, atlas, n_queries=50, n_jobs=1) -> tuple[FlipTransform, float]
predict_orientations(images, atlas) -> tuple[np.ndarray, list[dict[str, str]], float]
```

The old `pairwise_image_transform` survives only as the bootstrap for the running
mean, so it becomes the private `_bootstrap_mean_image`. The unused `errors`
return value of `running_mean_image_transform` is dropped.

### 3. Movie assembly

```python
orient_images(images, orientations) -> list[np.ndarray]
build_movie(images, shifts, alignment="center") -> np.ndarray
```

`build_movie` raises `ValueError` for an unknown alignment. Single-channel
straightened images are promoted to `(1, H, W)` so the movie axes are always
`TCYX`.

### 4. Pipeline block

```python
load_filemap(path) -> pl.DataFrame
load_orientation_cache(path) -> pl.DataFrame | None
predict_point_orientations(point, rows, atlas, channel) -> pl.DataFrame
write_point_movies(point, rows, columns, movie_dir, alignment, max_frames) -> None
get_args() -> argparse.Namespace
main() -> None
```

## Command line

Pipeline-contract flags, accepted in both modes:

- `--filemap` — pickled polars DataFrame (`.pkl`) or a filemap `.csv` / `.parquet`.
  The extension decides.
- `--config` — pickled pipeline config, optional. Supplies `experiment_dir`,
  `analysis_subdir`, `report_subdir`, `time_regex`, `point_regex`, `n_jobs`.
- `--block_config`, `--output` — accepted and ignored, so the pipeline's fixed
  command line is satisfied.
- `--n_jobs` — overrides the config value; defaults to `SLURM_CPUS_PER_TASK`, else 1.

Block options:

- `--straightened_column` — filemap column of straightened images used to predict
  orientation. Required.
- `--orientation_channel` — channel index within those images (default 0).
- `--atlas` — atlas CSV with an `atlas_image` column. Required unless the cache
  already covers every requested point.
- `--atlas_channel` — channel index within the atlas images (default 0).
- `--qc_column` — column to filter on; rows are kept where the value is `worm`.
  Default `None` keeps every row.
- `--movie_columns` — `All` (every column ending in `_str`) or a space-separated
  list of filemap columns. Default `All`. Independent of
  `--straightened_column`, which only selects the images orientation is
  predicted from.
- `--movie_dir` — default `<experiment_dir>/movies`. Movies are written to
  `<movie_dir>/<column basename>_movies/Point####_movie.tiff`, using
  `os.path.basename` rather than the old `lstrip("analysis/")`, which strips a
  character set rather than a prefix and mangles names like
  `analysis_sacha/...`.
- `--orientations` — default `<report_subdir>/orientations.csv`.
- `--recompute_orientations` — ignore an existing cache.
- `--alignment` — `center` or `left`. Default `center`.
- `--points` — `All`, or a space-separated list of point numbers and inclusive
  ranges (`3 10-20 42`). Default `All`.
- `--max_frames` — cap frames per movie. Default unset (all frames).

Alignment is applied *after* orientation correction: frames are first flipped so
that head is left and vulva up, then placed. `left` therefore anchors the head
end, whichever way the raw straightened image happened to face.

When `--config` is absent, `experiment_dir` and the report directory are derived
from the filemap path, so a manual run needs only `--filemap`, `--atlas` and
`--straightened_column`.

## Orientation cache

`orientations.csv` holds `Time, Point, head, vulva, head_confidence, shift_ax0,
shift_ax1`, one row per kept `(Time, Point)`. `head_confidence` is a per-point
value repeated across that point's rows, as in the current output. It lives in
the analysis report directory — where `ori_align.py`
wrote it, so existing experiments' files are picked up — but is never merged into
the filemap and never referenced by the pipeline.

On start, unless `--recompute_orientations` is set, the cache is read and a point
is considered covered when it appears in it at all. Only uncovered points are
predicted; results are merged with the cached rows and the file is rewritten.
This is what makes re-runs with a different `--alignment` or a different
`--movie_columns` cheap, since prediction dominates the runtime.

Coverage is per point rather than per row on purpose. A point whose series
contains a permanently unreadable frame has fewer cached rows than the filemap
has for it, and requiring every row to be present would re-predict that point on
every run without ever converging. The cost is that timepoints added to an
experiment after a run are not picked up for a point already in the cache;
`--recompute_orientations` is the answer there.

## Error handling

Following towbintools conventions at I/O boundaries:

- A straightened image that cannot be read, or whose requested channel is
  missing, is dropped from that point's series with a printed warning. A
  malformed frame must not cost the whole point — this already happens in
  practice on a small fraction of straightened images.
- A point that fails outright prints its traceback and is skipped; the remaining
  points continue. The run's exit status stays 0, as today.
- `ValueError` for invalid user-facing arguments: unknown alignment, missing
  columns, an atlas that is needed but not supplied.

Movies are written with `compression="zlib", imagej=True,
metadata={"axes": "TCYX"}`, unchanged.

## Pipeline patch

A branch in `~/towbintools_pipeline` adding a building block that produces no
output:

1. `BuildingBlock.run()` — an `else` branch for a `None` return type: pickle the
   block config and full config, build the command, submit it with a linker
   command that carries no result, return `None`. Today the method branches only
   on `"subdir"` and `"csv"`, falls through for anything else, and therefore
   neither runs the block nor chains to the next one.
2. `utils.create_linker_command` — omit `--result` when `result is None`, so the
   linker does not receive the literal string `"None"`.
   `block_linker.update_experiment_filemap` already falls through for an
   unrecognised return type and needs no change.
3. `CustomBuildingBlock.get_output_name` — return `None` for this case, and fix
   the existing branch chain, whose `if` / `if subdir is not None` / `elif`
   sequence raises `UnboundLocalError` for a `csv` block run against a subdir
   experiment.
4. A test alongside `tests/test_local_pipeline.py`: a custom block with a null
   return type that writes a sentinel file, followed by a second block, asserting
   both ran and that the filemap gained no column.

Commits follow the pipeline's guidelines: imperative mood, a `feat:` / `fix:`
prefix, body explaining why, no mention of AI tooling.

## Tests

`tests/test_straightened_movies.py`, plus `tests/README.md`, mirroring
`towbintools_pipeline/tests/`: pytest, `pytest.importorskip` for optional
dependencies, synthetic data generated at runtime, no bundled fixtures.

Geometry:

- `registration_origins` under `left` puts the minimum origin at 0 on the
  head-tail axis and reproduces `s_i - s_j` as the spacing between any two
  frames.
- `registration_origins` under `center` gives balanced margins for equally sized
  frames, and `center=True` keeps the registration origin at the canvas centre.
- `assemble_registered_stack` places each frame at its origin, returns the tight
  union canvas, and preserves dtype.
- `crop_to_mask` returns the mask's bounding box.

Orientation:

- `FlipTransform` composition, self-inverse, hashing, and the `head`/`vulva`
  labels for each of the four flips.
- `estimate_flip_transform` recovers the flip and the shift for a synthetic
  asymmetric blob that has been flipped and translated.
- `predict_orientations` on a synthetic worm series plus a matching atlas returns
  the correct `head`/`vulva` labels with confidence 1.0.

Movies:

- `build_movie` on a synthetic growing worm: under `left` the head column is
  stationary across frames; under `center` it migrates as the worm grows.
- `build_movie` raises `ValueError` on an unknown alignment.

End to end:

- A synthetic experiment (filemap, straightened TIFFs, atlas CSV) written to
  `tmp_path`, the script run as a subprocess in standalone mode, asserting the
  movie files exist with the expected shapes, the cache is written with the
  expected columns, and the filemap is unchanged on disk.
- A second run over the same directory reuses the cache: assert it does not
  rewrite the orientation values and that passing a different `--alignment`
  produces differently shaped movies.

## Repository layout

```
straightened_movies.py             the block
run_straightened_movies.slurm      standalone runner
configs/straightened_movies.yaml   example pipeline config using the custom block
tests/README.md
tests/test_straightened_movies.py
README.md                          rewritten from README.txt
legacy/                            the superseded scripts, kept for reference
```

`legacy/` holds `ori_align.py`, `movie_generation.py`,
`_debug_movie_generation.py`, the three `.slurm` files, `README.txt` and
`example_output.csv`.

## Style

Both repos' conventions: py39+ lowercase generics, `T | None` for optionals,
`np.ndarray` for images, `(..., H, W)` shapes, `UPPER_SNAKE_CASE` module
constants, docstrings in the towbintools `Parameters:` / `Returns:` / `Raises:`
format with `(default: value)` noted inline. Black at line length 88, isort with
`force_single_line`, flake8 ignoring E501/E203. No one-line wrapper functions and
no docstrings restating a self-evident signature.
