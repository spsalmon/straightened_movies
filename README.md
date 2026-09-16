# Straightened movies

Turns a set of straightened worm images into one movie per point: every frame is
flipped so the worm faces the same way, and registered against its neighbours so
the movie does not jitter.

Two stages, one script:

1. **Orientation.** For each point, all the images classified as worms are loaded
   and aligned relative to one another, so the whole timeseries faces one way —
   at this stage it is not known *which* way. The series is then compared against
   an atlas of manually oriented images, which acts as the datum for an absolute
   decision, and flipped to match it. Because the atlas convention is head
   **left** and vulva **up**, the orientation of each original image can be
   inferred and recorded.
2. **Movies.** The recorded orientations flip each frame to face head left and
   vulva up, and the registration shifts translate them so the worm does not jump
   between timepoints.

## Requirements

- Straightened images, and a quality-control column to filter out eggs and empty
  frames. Frames that slip through anyway are dropped: blank placeholders left by
  a failed straightening, and frames more than twice as thick or as long as their
  neighbours, or less than half. They get no orientation and stay out of the
  movies.
- An atlas: 50-100 manually oriented images is plenty. They need to cover the
  full range of morphology across the experiment, i.e. every developmental stage.
  Pre-made atlases live in `/mnt/towbin.data/shared/bgusev/atlas`.

## Running it

### Standalone

Edit the paths in `run_straightened_movies.slurm` and submit it, or call the
script directly:

```bash
~/.local/bin/micromamba run -n towbintools python straightened_movies.py \
    --filemap /path/to/analysis/report/analysis_filemap.csv \
    --straightened_column "analysis/ch1_ch2_raw_str" \
    --orientation_channel 1 \
    --qc_column "ch2_seg_str_qc" \
    --atlas /mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv \
    --atlas_channel 1 \
    --alignment left
```

### As a pipeline block

See `configs/straightened_movies.yaml`. The block is a `custom` building block
with `custom_script_return_type: null`, meaning it runs and hands over to the
next block but contributes nothing to the experiment filemap — its output is one
movie per point, which is not per-timepoint and has no place in the filemap.

That return type needs the pipeline to support a block with no output. See
`docs/superpowers/specs/2026-09-14-straightened-movies-rewrite-design.md`.

## Options

| Option | Meaning |
|---|---|
| `--filemap` | Pickled polars DataFrame (`.pkl`, what the pipeline passes) or a filemap `.csv` / `.parquet`. The extension decides. Required. |
| `--config` | Pickled pipeline config, optional. Supplies the experiment layout, the time/point regexes and `n_jobs`. |
| `--block_config`, `--output` | Accepted and ignored, so the pipeline's fixed command line is satisfied. |
| `--n_jobs` | Parallel workers. Defaults to the config value, then `SLURM_CPUS_PER_TASK`, then 1. |
| `--straightened_column` | Filemap column of straightened images used to predict orientation. Required. |
| `--orientation_channel` | Channel index **within those image files**. Default 0. |
| `--atlas` | Atlas CSV with an `atlas_image` column. Required unless the cache already covers every requested point. |
| `--atlas_channel` | Channel index within the atlas images. Default 0. |
| `--qc_column` | Column to filter on. Rows are kept where the value is `--qc_value`. Default: no filtering. |
| `--qc_value` | The label marking a usable worm. Default `worm`. |
| `--movie_columns` | `All` for every column ending in `_str`, or an explicit list. Default `All`. Independent of `--straightened_column`. |
| `--movie_dir` | Where movies go. Default `<experiment_dir>/movies`. |
| `--orientations` | The orientation cache. Default `<report_dir>/orientations.csv`. |
| `--recompute_orientations` | Ignore an existing cache and predict everything again. |
| `--alignment` | `center` or `left`. Default `center`. |
| `--points` | `All`, or point numbers and inclusive ranges: `--points 3 10-20 42`. Default `All`. |
| `--max_frames` | Cap the number of frames per movie. Default: all of them. |

### Picking an atlas and its channel

The atlas must be the same kind of microscopy image as what you are orienting —
a GFP body atlas for GFP body images, an mCherry pharynx atlas for mCherry
pharynx images. Prefer a channel with structures that are easy to orient, such as
germline or pharynx, over the worm body, which looks much the same either way
round.

Both channel options index **into the image file**, not into the experimental
channel numbering. An image from `analysis/ch2_raw_str` contains only one
channel, so its channel index is 0 even though the experimental channel is 2. The
two channels need to be the same kind of microscopy, but they are often different
integers.

## Alignment

`--alignment` sets how frames are placed on the movie canvas along the worm's
head-tail axis:

- **`center`** centres each frame, so the worm appears to grow in both
  directions. This is the historical behaviour.
- **`left`** anchors the head end, so growth runs left to right.

The dorsoventral axis is centred either way, and the registration shift on that
axis (`shift_ax0`) is not applied. Straightening already puts the midline on the
centre row of every frame (the segmentation's centre sits within 0.3 px of it on
real data), whereas the measured shift wandered by tens of pixels over a movie and
by about a pixel between frames, which showed as wobble and drift.

The choice of alignment **does not affect the registration**. Alignment only moves every frame by a common offset; the
spacing between frames, which is what removes the jitter, is the difference in
their registration shifts under either anchor. So the same orientation cache
serves both, and switching alignment only re-assembles the movies.

`left` anchors the head rather than pinning it rigidly at column zero: the frames
keep their registration offsets relative to one another, so the small
frame-to-frame jitter is still corrected.

## Output

```
<movie_dir>/
    ch1_ch2_raw_str_movies/
        Point0001_movie.tiff
        Point0002_movie.tiff
    ch2_seg_str_movies/
        ...
```

One directory per column in `--movie_columns`, created automatically. Movies are
written as zlib-compressed ImageJ TIFFs with axes `TCYX`.

## The orientation cache

`orientations.csv` is written to the analysis report directory with one row per
kept `(Time, Point)`:

| Column | Meaning |
|---|---|
| `Time`, `Point` | indices |
| `head` | `L` or `R`: where the head faces in the original straightened image |
| `vulva` | `U` or `D`: where the vulva is, or would be |
| `head_confidence` | proportion of atlas comparisons that agreed on the head direction, repeated across the point's rows |
| `shift_ax0`, `shift_ax1` | the translations that best register consecutive frames |

It is a cache, not a report: **it is never added to the experiment filemap**. On
the next run over the same experiment, any point already in it is skipped, which
is what makes re-running with a different `--alignment` or `--movie_columns`
cheap — prediction dominates the runtime. Use `--recompute_orientations` to
rebuild it.

### How far to trust the labels

`head_confidence` is the useful number: it is the share of atlas comparisons that
agreed, so a point well below 1.0 is worth looking at.

`vulva` is **not reliable**, for three reasons:

- Worms are close to rotationally symmetric, so movies may show random up-down
  flips.
- A worm imaged halfway through a rotation gives a smooth U-to-D transition
  rather than a clean label.
- The vulva may not be visible at all in the channel being used — pharynx images,
  or male worms. The label is then only an inferred location, used to
  disambiguate the up-down axis.

## Layout

```
straightened_movies.py             the block
run_straightened_movies.slurm      standalone runner
configs/straightened_movies.yaml   example pipeline config
tests/                             pytest suite, synthetic data only
docs/superpowers/                  design spec and implementation plan
legacy/                            the pre-rewrite scripts, kept for reference
```

`straightened_movies.py` is organised in four sections. The first, *Generic
helpers*, depends on nothing below it and is written in `towbintools` style so it
can be lifted into the library as-is.

## Tests

```bash
~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v
```
