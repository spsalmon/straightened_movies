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

# EXPERIMENT_DIR = Path("/mnt/towbin.data/shared/USER/EXPERIMENT")

FILEMAP = Path(
    "/mnt/towbin.data/shared/plenart/20242901_CREST_10X_wBT318_timings_includes_smaller_chambers_4-72h/analysis/report/analysis_filemap.csv"
)
ORIENTATIONS_CSV = Path(
    "/mnt/towbin.data/shared/plenart/20242901_CREST_10X_wBT318_timings_includes_smaller_chambers_4-72h/analysis/report/new_orientations.csv"
)

# "All" or list of specific str image columns in filemap e.g. ["analysis/ch1_ch2_raw_str", "analysis/ch1_seg_str"]
# "All" assumes columns of interest end with "_str" <- this excludes columns like "ch1_seg_str_volume"
STR_IMG_COLS = "All"
MOVIE_DIR = Path("/mnt/towbin.data/shared/bgusev/general_orientation/_debug_movies")
# Subdirectories will be automatically created for each column in STR_IMG_COLS, e.g. MOVIE_DIR/"ch1_ch2_raw_str_movies" and MOVIE_DIR/"ch1_seg_str_movies"

# Which points to process. Either "All" or a list of points e.g. [10, 15, 200]. If only one point, still use a list: [10].
# Can also use range(1, 100) to process points 1 to 99 (python range is exclusive on upper bound)
WHICH_POINTS = [4]

# automatically pull the number of CPUs allocated to SBATCH job. Default to 1 if run outside a job
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


##########
# SCRIPT #
##########

Image = npt.NDArray[np.number]
shift = npt.NDArray[np.integer]


def normalise_shifts(shifts: list[shift]) -> list[shift]:
    # subtract the mean 'drift' so that all images are roughly centered
    shifts = np.array(shifts)
    drift = np.round(shifts.mean(axis=0)).astype(int)
    norm_shifts = shifts - drift
    return norm_shifts


def pad_to_shape(image: Image, shape: tuple[int, ...]) -> Image:
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
    images: list[Image], shifts: list[shift], channel_axis: int = 0
) -> Image:
    shifts = normalise_shifts(shifts)
    max_shape = np.array([img.shape for img in images]).max(axis=0)
    # give some space for images to shift around
    target_shape = max_shape * 2
    # don't pad the channel axis
    target_shape[channel_axis] = max_shape[channel_axis]
    movie_mask = [np.ones_like(img, dtype=bool) for img in images]
    movie_mask = np.array([pad_to_shape(mask, target_shape) for mask in movie_mask])
    movie = np.array([pad_to_shape(img, target_shape) for img in images])
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


def process_point(point, point_df, str_img_cols):
    point_df = point_df.sort_values(by="Time")
    for c in str_img_cols:
        dir_name = c.lstrip("analysis/") + "_movies"
        movie_dir = MOVIE_DIR / dir_name
        movie_dir.mkdir(exist_ok=True, parents=True)
        try:
            images = [tifffile.imread(f) for f in point_df[c]]
            images = [atleast_1_channel(img) for img in images]
            images = correct_orientations(images, point_df[["head", "vulva"]])
            shifts = point_df[["shift_ax0", "shift_ax1"]].values
            movie = create_movie(images, shifts)
            tifffile.imwrite(movie_dir / f"Point{point:04}_movie.tiff", movie)
        except Exception:
            print(f"Error creating {c} movie for point {point}")
            print(traceback.format_exc())


def main():
    global STR_IMG_COLS, ORIENTATIONS_CSV, FILEMAP, WHICH_POINTS

    FILEMAP = Path(FILEMAP)
    ORIENTATIONS_CSV = Path(ORIENTATIONS_CSV)

    filemap_ori = load_filemap_ori(FILEMAP, ORIENTATIONS_CSV)
    if STR_IMG_COLS == "All":
        STR_IMG_COLS = _guess_str_img_cols(filemap_ori)

    t_start = time.time()

    print(
        f"{time.asctime()} - Making movies for the following filemap columns: {STR_IMG_COLS}"
    )

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
        delayed(process_point)(point, point_df, STR_IMG_COLS)
        for point, point_df in groupby
    )
    _ = list(tqdm(job_iter, total=len(groupby), desc="Creating movies for points: "))

    t_end = time.time()
    print(f"{time.asctime()} - Finished creating movies")
    mins_duration = int(round((t_end - t_start) / 60))
    print(f"Time taken: {mins_duration}min")


if __name__ == "__main__":
    main()
