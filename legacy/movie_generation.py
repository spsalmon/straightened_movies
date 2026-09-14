import os
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from tqdm import tqdm

# REQUIREMENTS:
#   - straightened images
#   - ori_align.py result


##########
# CONFIG #
##########

EXPERIMENT_DIR = Path("/mnt/towbin.data/shared/kstojanovski/20240202_Orca_10x_yap-1del_col-10-tir_wBT160-186-310-337-380-393_25C_20240202_171239_051")

FILEMAP = EXPERIMENT_DIR / "analysis_sacha" / "report" / "analysis_filemap.csv"
ORIENTATIONS_CSV = EXPERIMENT_DIR / "analysis_sacha" / "report" / "orientations.csv"

# "All" or list of specific str image columns in filemap e.g. ["analysis/ch1_ch2_raw_str", "analysis/ch1_seg_str"]
# "All" assumes columns of interest end with "_str" <- this excludes columns like "ch1_seg_str_volume"
STR_IMG_COLS = ["analysis_sacha/ch1_ch2_raw_str"]
MOVIE_DIR = EXPERIMENT_DIR / "movies"
# Subdirectories will be automatically created for each column in STR_IMG_COLS, e.g. MOVIE_DIR/"ch1_ch2_raw_str_movies" and MOVIE_DIR/"ch1_seg_str_movies"

# How frames are aligned within the movie canvas along the worm's head->tail axis:
#   "center" - frames are centred, so growth spreads out in both directions
#   "left"   - the head end is pinned to the left edge, so all growth extends to the right
# The dorsoventral axis stays centred either way.
ALIGNMENT = "left"

# Which points to process. Either "All" or a list of points e.g. [10, 15, 200]. If only one point, still use a list: [10].
# Can also use range(1, 100) to process points 1 to 99 (python range is exclusive on upper bound)
WHICH_POINTS = "All"

# automatically pull the number of CPUs allocated to SBATCH job. Default to 1 if run outside a job
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
MAX_FRAMES = 250

##########
# SCRIPT #
##########

Image = npt.NDArray[np.number]
shift = npt.NDArray[np.integer]

ALIGNMENTS = ("center", "left")
# axis of the images holding the worm's head->tail length, counted from the end
# so that it indexes both the (..., ax0, ax1) shift vectors and the padded movie frames
LENGTH_AXIS = -1


def normalise_shifts(
    shifts: list[shift], flush_axes: tuple[int, ...] = ()
) -> list[shift]:
    # subtract the mean 'drift' so that all images are roughly centered
    shifts = np.array(shifts)
    drift = np.round(shifts.mean(axis=0)).astype(int)
    # for flush_axes, subtract the smallest shift instead of the mean, so that every
    # shift along them is >= 0. This keeps the images registered to each other while
    # pinning them to the low-index edge, with no np.roll wrap-around
    for ax in set(ax % shifts.shape[-1] for ax in flush_axes):
        drift[ax] = shifts[:, ax].min()
    norm_shifts = shifts - drift
    return norm_shifts


def pad_to_shape(
    image: Image, shape: tuple[int, ...], flush_axes: tuple[int, ...] = ()
) -> Image:
    img_shape = np.array(image.shape)
    target_shape = np.array(shape)
    if len(img_shape) != len(target_shape):
        raise ValueError(
            f"Image shape {img_shape} and target shape {target_shape} must have the same number of dimensions"
        )

    pad = target_shape - img_shape
    if (pad < 0).any():
        raise ValueError(
            f"Target shape for padding must be larger than image shape. image.shape: {image.shape}; target_shape: {target_shape}"
        )

    before_pad = pad // 2
    # if pad is odd, after_pad will be 1 more than before_pad
    after_pad = pad - before_pad
    # flush_axes are padded on one side only, leaving the image against the low-index edge
    for ax in set(ax % image.ndim for ax in flush_axes):
        before_pad[ax] = 0
        after_pad[ax] = pad[ax]
    # np.pad takes pad in form [(before_ax0, after_ax0), (before_ax1, after_ax1), ...]
    pad = list(zip(before_pad, after_pad))
    return np.pad(image, pad, mode="constant")

def rectangular_crop(
    ndimage: Image,
    keep_mask: Image,
    symmetric: bool = False,
    axes: Optional[tuple[int, ...]] = None,
) -> Image:
    if ndimage.shape != keep_mask.shape:
        raise ValueError(
            f"Image shape {ndimage.shape} and keep_mask shape {keep_mask.shape} must be the same"
        )
    all_axes = np.arange(ndimage.ndim)
    if axes is None:
        axes = set(all_axes)
    else:
        # modulus ndim to allow for -1, -2 arguments
        axes = set(ax % ndimage.ndim for ax in axes)
    axis_keep_arrays = []
    for axis in range(ndimage.ndim):
        other_axes = tuple(np.delete(all_axes, axis))
        # axis_mask.shape is now ndimage.shape[axis]
        axis_mask = keep_mask.any(other_axes)
        if axis not in axes:
            # keep everything if axis is not in axes
            axis_keep_arrays.append(np.ones_like(axis_mask))
            continue
        start_keep = np.argmax(axis_mask)
        end_keep = np.argmax(axis_mask[::-1])
        if symmetric:
            crop_amount = min(start_keep, end_keep)  # type: ignore # np.argmax returns type 'intp' which is not recognised by min as simply being an int
            start_keep, end_keep = crop_amount, crop_amount
        # convert end_keep to index into axis_mask rather than into axis_mask[::-1]
        end_keep = len(axis_mask) - end_keep

        axis_keep = np.zeros_like(axis_mask)
        # I love 0-start indexing and exclusive upper bounds :D
        axis_keep[start_keep:end_keep] = True
        axis_keep_arrays.append(axis_keep)

    keep_indices = np.ix_(*axis_keep_arrays)
    cropped_ndimage = ndimage[keep_indices]
    return cropped_ndimage


def create_movie(
    images: list[Image],
    shifts: list[shift],
    channel_axis: int = 0,
    alignment: str = ALIGNMENT,
) -> Image:
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {alignment!r}")
    # "left" keeps the head end against the left edge of the canvas so that the worm
    # only ever grows rightwards. The other axes stay centered
    flush_axes = (LENGTH_AXIS,) if alignment == "left" else ()
    shifts = normalise_shifts(shifts, flush_axes=flush_axes)
    max_shape = np.array([img.shape for img in images]).max(axis=0)
    # give some space for images to shift around
    target_shape = max_shape * 2
    # don't pad the channel axis
    target_shape[channel_axis] = max_shape[channel_axis]
    movie_mask = [np.ones_like(img, dtype=bool) for img in images]
    movie_mask = np.array(
        [pad_to_shape(mask, target_shape, flush_axes=flush_axes) for mask in movie_mask]
    )
    movie = np.array(
        [pad_to_shape(img, target_shape, flush_axes=flush_axes) for img in images]
    )
    for i in range(len(movie)):
        movie[i] = np.roll(movie[i], shifts[i], axis=(-2, -1))
        movie_mask[i] = np.roll(movie_mask[i], shifts[i], axis=(-2, -1))
    # crop black borders
    # explicit "movie_mask" rather than "movie > 0" to make sure that for a single Point, movies for different channel/seg images are the same shape (assuming that images are the same shape for each TimePoint to begin with)
    movie = rectangular_crop(movie, keep_mask=movie_mask, axes=(-2, -1))
    return movie


def load_filemap_ori(filemap, ori_csv):
    filemap_df = pd.read_csv(filemap)
    ori_df = pd.read_csv(ori_csv)
    combined = filemap_df.merge(ori_df, on=["Time", "Point"], how="right")
    combined = combined.sort_values(by=["Point", "Time"])
    return combined


def _guess_str_img_cols(df):
    all_cols = df.columns
    str_img_cols = [c for c in all_cols if c.endswith("_str")]
    return str_img_cols


def atleast_1_channel(image: Image) -> Image:
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    return image


def correct_orientations(
    images: list[Image], orientations: pd.DataFrame
) -> list[Image]:
    flipped_images = []
    for (_, row), image in zip(orientations.iterrows(), images):
        if row.loc["head"] == "R":
            image = np.flip(image, axis=-1)
        if row.loc["vulva"] == "D":
            image = np.flip(image, axis=-2)
        flipped_images.append(image)
    return flipped_images


def process_point(point, point_df, str_img_cols, max_frames=MAX_FRAMES):
    point_df = point_df.sort_values(by="Time")
    for c in str_img_cols:
        dir_name = c.lstrip("analysis/") + "_movies"
        movie_dir = MOVIE_DIR / dir_name
        movie_dir.mkdir(exist_ok=True, parents=True)
        try:
            images = [tifffile.imread(f) for f in point_df[c]][0:max_frames]
            images = [atleast_1_channel(img) for img in images]
            images = correct_orientations(images, point_df[["head", "vulva"]])
            shifts = point_df[["shift_ax0", "shift_ax1"]].values
            movie = create_movie(images, shifts)
            tifffile.imwrite(movie_dir / f"Point{point:04}_movie.tiff", movie, compression="zlib", imagej=True, metadata={"axes": "TCYX"})
        except Exception:
            print(f"Error creating {c} movie for point {point}")
            print(traceback.format_exc())


def main():
    global STR_IMG_COLS, ORIENTATIONS_CSV, FILEMAP, WHICH_POINTS

    FILEMAP = Path(FILEMAP)
    ORIENTATIONS_CSV = Path(ORIENTATIONS_CSV)

    if ALIGNMENT not in ALIGNMENTS:
        raise ValueError(f"ALIGNMENT must be one of {ALIGNMENTS}, got {ALIGNMENT!r}")

    filemap_ori = load_filemap_ori(FILEMAP, ORIENTATIONS_CSV)
    if STR_IMG_COLS == "All":
        STR_IMG_COLS = _guess_str_img_cols(filemap_ori)

    t_start = time.time()

    print(
        f"{time.asctime()} - Making movies for the following filemap columns: {STR_IMG_COLS}"
    )
    print(f"Frames will be {ALIGNMENT}-aligned")

    groupby = filemap_ori.groupby("Point")
    groupby = [(point, point_df) for point, point_df in groupby]

    if WHICH_POINTS != "All":
        if isinstance(WHICH_POINTS, int):
            WHICH_POINTS = [WHICH_POINTS]
        WHICH_POINTS = set(WHICH_POINTS)
        groupby = [
            (point, point_df) for point, point_df in groupby if point in WHICH_POINTS
        ]

    print(
        f"Movies will be created for points: {sorted([point for point, _ in groupby])}"
    )

    job_iter = Parallel(n_jobs=THREADS, return_as="generator")(
        delayed(process_point)(point, point_df, STR_IMG_COLS, max_frames=MAX_FRAMES)
        for point, point_df in groupby
    )
    _ = list(tqdm(job_iter, total=len(groupby), desc="Creating movies for points: "))

    t_end = time.time()
    print(f"{time.asctime()} - Finished creating movies")
    mins_duration = int(round((t_end - t_start) / 60))
    print(f"Time taken: {mins_duration}min")


if __name__ == "__main__":
    main()
