import os
import time
import traceback
from itertools import compress, product
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from skimage import exposure, metrics, registration, transform  # type: ignore
from tqdm import tqdm

# REQUIREMENTS:
#   - straightened images
#   - worm type classification
#   - manually flipped atlas images (50-100 is enough, even less could be fine, but not tested. They just need to be representative of the full morphology of the worm throughout the experiment i.e. all stages of development)


##########
# CONFIG #
##########
# link the full path to the experiment here. Double check the other paths make sense, or can overwrite them manually for custom read/save paths
# e.g. Path("/mnt/towbin.data/shared/EXPERIMENT")
EXPERIMENT_DIR = Path("/mnt/towbin.data/shared/kstojanovski/20240202_Orca_10x_yap-1del_col-10-tir_wBT160-186-310-337-380-393_25C_20240202_171239_051")
# default FILEMAP location is: EXPERIMENT_DIR / "analysis" / "report" / "analysis_filemap.csv"
FILEMAP = EXPERIMENT_DIR / "analysis_sacha" / "report" / "analysis_filemap.csv"
# column in filemap with raw straightened images, e.g. "analysis/ch1_ch2_raw_str"
RAW_STR_COL = "analysis_sacha/ch1_ch2_raw_str"
# column in filemap with "worm type" for filtering away eggs and empty images, e.g. "ch2_seg_str_worm_type"
WORM_TYPE_COL = "ch2_seg_str_worm_type"
# channel used for predicting orientation. given an option, prefer a channel with easy to orient structures, e.g. germline or pharynx, rather than the worm body which looks similar regardless of orientation
# this is a channel into the image, not necessarily a channel in the experiment
RAW_STR_CHANNEL = 1
# can be a specific full path e.g. Path("/mnt/towbin.data/shared/EXPERIMENT/analysis/report/orientation.csv")
# default output is: EXPERIMENT_DIR / "analysis" / "report" / "orientations.csv"
OUTPUT_FILE = EXPERIMENT_DIR / "analysis_sacha" / "report" / "orientations.csv"

# Atlas is the set of manually oriented images that are used to determine absolute orientation of the images
# check the "/mnt/towbin.data/shared/bgusev/atlas" directory for the atlases I created
ATLAS_CSV = Path("/mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv")
# atlas image channel needs to correspond to raw_str channel
# this is a channel into the image, not necessarily a channel in the experiment
ATLAS_CHANNEL = 1


#########
# OTHER #
#########
# by default reads the number of CPUS assigned to the slurm job, but can be overwritten manually. Defaults to 1 if not run inside a slurm job
# THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
THREADS = 4


#############
# ALGORITHM #
#############


Image = npt.NDArray[np.number]
shift = npt.NDArray[np.integer]
T = TypeVar("T")
kw = dict[str, Any]


class FlipTransformation:
    def __init__(self, mirror_axes: Optional[tuple[int, ...]] = None):
        if mirror_axes is None:
            mirror_axes = tuple()
        # frozen set to allow hashing
        self.mirror_axes = frozenset(mirror_axes)

    def __call__(self, image: Image) -> Image:
        return np.flip(image, axis=tuple(self.mirror_axes))

    def __add__(self, other: "FlipTransformation") -> "FlipTransformation":
        self_axes = self.mirror_axes
        other_axes = other.mirror_axes
        new_axes = self_axes.symmetric_difference(other_axes)
        return FlipTransformation(tuple(new_axes))

    def __hash__(self) -> int:
        # implementing __hash__ and __eq__ to allow for tf1==tf2 comparison and usage as keys in a dict
        return hash(self.mirror_axes)

    def __eq__(self, other: "FlipTransformation") -> bool:
        # implementing __hash__ and __eq__ to allow for tf1==tf2 comparison and usage as keys in a dict
        return self.mirror_axes == other.mirror_axes

    def __str__(self) -> str:
        return f"FlipTransformation(mirror_axes={tuple(self.mirror_axes)})"

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, axis: int) -> bool:
        # e.g. for testing "if 1 in tf: ...; else: ..."
        return axis in self.mirror_axes

    def inferred_orientation(self) -> dict[str, str]:
        # Head Left, Vulva Up is the default orientation. This method returns the inferred orientation based on the flip transformation
        result = {
            "head": "L" if 1 not in self else "R",
            "vulva": "U" if 0 not in self else "D",
        }
        return result

    @property
    def inverse(self) -> "FlipTransformation":
        # inverse of a flip is itself
        return self

    @classmethod
    def estimate_transform(
        cls,
        reference: Image,
        moving: Image,
        channel_axis: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> tuple["FlipTransformation", shift, float]:
        """For all possible shifts across all axes, find the best FlipTranform such that reference and FlipTransform(moving) are most similar.

        Transformation error is calculated by registering the moving image onto reference using phase cross correlation
        and then calculating the dissimilarity between the registered moving image and the reference, whilst also penalising too much movement.

        If scale<1 is provided, the images are rescaled to speed up computation. The scale is then accounted for in the shift output.
        """
        if (scale is not None) and (scale < 1):
            # allow rescaling to speed up computation for larger images. For larger images, there is little accuracy loss
            reference = transform.rescale(
                reference, scale, anti_aliasing=True, channel_axis=channel_axis
            )
            moving = transform.rescale(
                moving, scale, anti_aliasing=True, channel_axis=channel_axis
            )
        shape_thresh = np.round(
            np.abs(np.array(reference.shape) - np.array(moving.shape)) / 2
        )
        common_shape = np.max(np.vstack([reference.shape, moving.shape]), axis=0)
        # print(f"shape_delta: {shape_thresh}, common_shape: {common_shape}")
        possible_transforms = cls.all_transforms(
            reference.ndim, channel_axis=channel_axis
        )
        shifts = []
        errors = []
        for tf in possible_transforms:
            tf_moving = tf(moving)
            shift, dissimilarity = optimal_translation(reference, tf_moving)
            # don't penalise shifts less than thresh. this allows image to move sufficiently to account for shape mismatch
            shift_error = np.abs(shift) - shape_thresh
            shift_error[shift_error < 0] = 0
            shift_error = (shift_error / common_shape.max()).sum() * 2
            # print(f"{tf}, {shift}, {dissimilarity}, {shift_error}")
            # total error is a combination of dissimilarity and shift error. This accounts for both the quality of the match and the shift required to achieve the match
            total_error = dissimilarity + shift_error
            shifts.append(shift)
            errors.append(total_error)

        tf_out = possible_transforms[np.argmin(errors)]
        error_out = np.min(errors)
        shift_out = shifts[np.argmin(errors)].astype(int)
        # print(shift_out, scale, np.round(shift_out / scale).astype(int))
        if scale is not None and scale < 1:
            shift_out = shift_out / scale
            shift_out = np.round(shift_out).astype(int)
        return tf_out, shift_out, error_out

    @staticmethod
    def all_transforms(
        ndim: int, channel_axis: Optional[int] = None
    ) -> list["FlipTransformation"]:
        axes = list(range(ndim))
        if channel_axis is not None:
            axes.pop(channel_axis)
        # include is a list of tuples of all possible combinations of True and False across axes
        include = product((False, True), repeat=len(axes))
        flip_axes = [tuple(compress(axes, inc)) for inc in include]
        transforms = [FlipTransformation(axes) for axes in flip_axes]
        return transforms


def structural_dissimilarity(
    img1: Image, img2: Image, channel_axis: Optional[int] = None
) -> float:
    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same shape - img1: {img1.shape}, img2: {img2.shape}"
        )
    # default skimage winsize is 7, but 21 chosen because it seems to perform better for flip estimation
    # because the longer range of the window is more forgiving of images with big morphological difference or slight channel misalignment
    # this results in a dissimilarity measure that has a bigger dynamic range, making it easier to pick a better alignment. (e.g. 0.10001vs 0.10010 and 0.1 vs 0.2 dissimilarities)
    # in cases when image can't fit a 21x21 window, winsize is set to the minimum of 21 and the smallest axis length
    winsize = min(21, *img1.shape)
    # winsize must be odd
    winsize = winsize - ((winsize + 1) % 2)
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    similarity = metrics.structural_similarity(
        img1,
        img2,
        data_range=data_range,
        win_size=winsize,
        channel_axis=channel_axis,
        gaussian_weights=False,
        use_sample_covariance=False,
    )
    dissimilarity = 1 - similarity
    return dissimilarity


def optimal_translation(reference: Image, moving: Image) -> tuple[shift, float]:
    # pad to avoid FFT edge effects due to circular convolution
    target_shape = tuple(np.array(reference.shape) + np.array(moving.shape) - 1)

    ref_padded = pad_to_shape(reference, target_shape)
    ref_mask_padded = pad_to_shape(np.ones_like(reference, dtype=bool), target_shape)

    mov_padded = pad_to_shape(moving, target_shape)
    mov_mask_padded = pad_to_shape(np.ones_like(moving, dtype=bool), target_shape)
    # for masked phase cross correlation, the _ are None
    shift, _, _ = registration.phase_cross_correlation(
        reference_image=ref_padded,
        reference_mask=ref_mask_padded,
        moving_image=mov_padded,
        moving_mask=mov_mask_padded,
        overlap_ratio=0.9,
        normalization=None,
    )
    # cast to int because masked phase cross correlation is always int, but the return type is float
    # int type is needed for use in np.roll
    shift = shift.astype(int)
    mov_padded = np.roll(mov_padded, shift)
    mov_mask_padded = np.roll(mov_mask_padded, shift)

    overlap_mask = ref_mask_padded & mov_mask_padded  # type:ignore # for some reason np.roll does weird things with typing
    ref_overlap_region = rectangular_crop(ref_padded, overlap_mask)
    mov_overlap_region = rectangular_crop(mov_padded, overlap_mask)
    dissimilarity = structural_dissimilarity(
        ref_overlap_region, mov_overlap_region, channel_axis=None
    )

    return shift, dissimilarity


# def translate_and_stack(images):
#     args = [
#         {"reference": ref, "moving": mov} for ref, mov in zip(images[:-1], images[1:])
#     ]
#     pair_wise_shifts = parallel_job(
#         optimal_translation,
#         args,
#         tqdm_kwargs={
#             "desc": "Optimal Translation",
#             "total": len(args),
#         },
#     )
#     pair_wise_shifts = np.array(pair_wise_shifts)
#     # add the zero shift for the first image as len(pair_wise_shifts) = len(images) - 1
#     pair_wise_shifts = np.r_[[[0, 0]], pair_wise_shifts]
#     # cumsum shift for global shift rather than local pair-wise shift
#     pair_wise_shifts = np.cumsum(pair_wise_shifts, axis=0)
#     pair_wise_shifts = normalise_shifts(pair_wise_shifts)

#     image_shapes = np.array([img.shape for img in images])
#     max_shape = image_shapes.max(axis=0)
#     target_shape = max_shape * 2

#     padded_images = [pad_to_shape(img, target_shape) for img in images]

#     shifted_images = [
#         np.roll(img, shift, axis=(0, 1))
#         for img, shift in zip(padded_images, pair_wise_shifts)
#     ]
#     stack = np.stack(shifted_images, axis=0)
#     # crop black borders
#     stack = rectangular_crop(stack, stack > 0, axes=(1, 2))
#     return stack


def normalise_shifts(shifts: list[shift]) -> list[shift]:
    # subtract the mean 'drift' so that all images are roughly centered
    shifts = np.array(shifts)
    drift = np.round(shifts.mean(axis=0)).astype(int)
    norm_shifts = shifts - drift
    return norm_shifts


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
        axes = all_axes.copy()
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

def closest_shape_match(query_image: Image, reference_list: list[Image]) -> Image:
    reference_shapes = np.array([img.shape for img in reference_list])
    # newaxis for correct broadcasting
    shape_delta = reference_shapes - np.array(query_image.shape)[np.newaxis, ...]
    shape_delta = np.abs(shape_delta).sum(axis=1)
    best_shape_match = np.argmin(shape_delta)
    return reference_list[best_shape_match]


def create_movie(images: list[Image], shifts: list[shift]) -> Image:
    shifts = normalise_shifts(shifts)
    max_shape = np.array([img.shape for img in images]).max(axis=0)
    # give some space for images to shift around
    target_shape = max_shape * 2
    movie = np.array([pad_to_shape(img, target_shape) for img in images])
    for i in range(len(movie)):
        movie[i] = np.roll(movie[i], shifts[i], axis=(0, 1))
    # crop black borders
    movie = rectangular_crop(movie, keep_mask=movie > 0, axes=(1, 2))
    return movie


def _calc_auto_scale(img: Image, size_threshold: int = 40_000) -> float:
    if img.size < size_threshold:
        return 1.0
    # scale is applied to axis lengths, so img.size scales with scale ^ ndim
    # TODO: does not respect channels
    scale = (size_threshold / img.size) ** (1 / img.ndim)
    return scale


def pairwise_image_transform(
    images: list[Image], channel_axis: Optional[int] = None, n_jobs: int = THREADS
) -> tuple[list[Image], list[FlipTransformation], list[shift], list[float]]:
    if len(images) == 1:
        return (
            images,
            [FlipTransformation()],
            np.array([0, 0]).reshape((1, 2)),
            np.array([0]),
        )
    image_pairs = list(zip(images[:-1], images[1:]))
    args = [
        {
            "reference": ref,
            "moving": mov,
            "channel_axis": channel_axis,
            "scale": min(_calc_auto_scale(ref), _calc_auto_scale(mov)),
        }
        for ref, mov in image_pairs
    ]
    result = parallel_job(
        func=FlipTransformation.estimate_transform,
        func_args=args,
        n_jobs=n_jobs,
        tqdm_kwargs={
            "desc": "Estimating Pairwise Transform",
            "total": len(args),
            "disable": True,
        },
    )
    transforms, shifts, errors = zip(*result)
    cumulative_transform = [FlipTransformation(mirror_axes=None)]
    cumulative_shift = [np.array([0, 0])]
    for tf, s in zip(transforms, shifts):
        prev_tf = cumulative_transform[-1]
        net_tf = prev_tf + tf

        if 0 in prev_tf:
            s[0] *= -1
        if 1 in prev_tf:
            s[1] *= -1
        cumulative_transform.append(net_tf)
        cumulative_shift.append(cumulative_shift[-1] + s)

    tf_imgs = [tf(i) for tf, i in zip(cumulative_transform, images)]
    errors = np.array(errors)
    cumulative_shift = np.array(cumulative_shift)
    return (
        tf_imgs,
        cumulative_transform,
        cumulative_shift,
        errors,
    )


def running_mean_image_transform(
    images: list[Image],
    running_mean_window: int,
    channel_axis: Optional[tuple[int, ...]] = None,
) -> tuple[list[Image], list[FlipTransformation], list[shift], list[float]]:
    _tf_imgs, _, _shifts, _ = pairwise_image_transform(
        images[:running_mean_window], channel_axis=channel_axis, n_jobs=1
    )
    bootstrap_mean_img = _mean_image(_tf_imgs, _shifts)

    tfs = []
    tf_imgs = [bootstrap_mean_img] * running_mean_window
    shifts = [np.array([0, 0])] * running_mean_window
    errors = []
    _mean_images = []
    for img in images:
        mean_image = _mean_image(
            tf_imgs[-running_mean_window:], shifts[-running_mean_window:]
        )
        scale = min(_calc_auto_scale(mean_image), _calc_auto_scale(img))
        tf, shift, e = FlipTransformation.estimate_transform(
            mean_image, img, channel_axis=channel_axis, scale=scale
        )
        tfs.append(tf)
        tf_imgs.append(tf(img))
        shifts.append(shift)
        errors.append(e)
        _mean_images.append(mean_image)

    tf_imgs = tf_imgs[running_mean_window:]
    shifts = np.array(shifts[running_mean_window:])
    return (
        tf_imgs,
        tfs,
        shifts,
        errors,
        # _mean_images, # for debugging in notebook
    )


def _mean_image(images: list[Image], shifts: list[shift]) -> Image:
    shifts = normalise_shifts(shifts)
    max_shape = np.array([img.shape for img in images]).max(axis=0)
    # give some space for images to shift around
    target_shape = max_shape * 2
    masks = [np.ones_like(img, dtype=bool) for img in images]
    masks = np.array([pad_to_shape(m, target_shape) for m in masks])
    images = np.array([pad_to_shape(img, target_shape) for img in images])
    for i in range(len(images)):
        images[i] = np.roll(images[i], shifts[i], axis=(0, 1))
        masks[i] = np.roll(masks[i], shifts[i], axis=(0, 1))
    flat_mask = masks.any(0)
    mean_image = images.mean(0)
    # symmetric crop is important to keep the centre of the mean image in the centre of the cropped array
    # since cross correlation shifts are relative to image centre
    mean_image = rectangular_crop(mean_image, keep_mask=flat_mask, symmetric=True)
    return mean_image


def consensus_align_to_atlas(
    images: list[Image], atlas: list[Image], num_queries=50, n_jobs=1, tqdm_disable=True
) -> tuple[FlipTransformation, float]:
    # use roughly num_queries evenly spaced  images to estimate the consensus transform
    interval = max(
        len(images) // num_queries, 1
    )  # if len(images) < num_queries, interval = 1
    query_imgs = images[::interval]
    # for each query, find the closest_shape_match in atlas
    # MAYBE: use a more sophisticated image retrieval method, as using image shape to approximate a developmental stage match is not great
    reference_imgs = [closest_shape_match(q, atlas) for q in query_imgs]

    args = [
        {
            "reference": r,
            "moving": q,
            "scale": min(_calc_auto_scale(r), _calc_auto_scale(q)),
        }
        for q, r in zip(query_imgs, reference_imgs)
    ]
    result = parallel_job(
        FlipTransformation.estimate_transform,
        args,
        n_jobs=n_jobs,
        tqdm_kwargs={
            "desc": "Aligning to Atlas",
            "total": len(args),
            "disable": tqdm_disable,
        },
    )

    # groupby without pandas :(
    summary = dict()
    for tf, shift, e in result:
        if tf not in summary:
            summary[tf] = []
        summary[tf].append(e)

    length_axis = np.arange(images[0].ndim)[-1]

    length_flip_summary = {tf: e for tf, e in summary.items() if length_axis in tf}
    length_flip_n = sum(len(e) for e in length_flip_summary.values())
    length_keep_summary = {tf: e for tf, e in summary.items() if length_axis not in tf}
    length_same_n = sum(len(e) for e in length_keep_summary.values())

    # print([f"{k.mirror_axes}, {len(v)}" for k, v in length_flip_summary.items()], "\n")
    # print([f"{k.mirror_axes}, {len(v)}" for k, v in length_keep_summary.items()], "\n")
    if length_flip_n > length_same_n:
        ret_tf = max(
            length_flip_summary.keys(), key=lambda k: len(length_flip_summary[k])
        )
        head_consensus = length_flip_n / (length_flip_n + length_same_n)
    else:
        ret_tf = max(
            length_keep_summary.keys(), key=lambda k: len(length_keep_summary[k])
        )
        head_consensus = length_same_n / (length_flip_n + length_same_n)
    return ret_tf, head_consensus


def parallel_job(
    func: Callable[..., T],
    func_args: list[kw],
    n_jobs=THREADS,
    timeout=None,
    tqdm_kwargs: Optional[dict[str, Any]] = None,
) -> list[T]:
    # correct typing of this function would require python 3.11 for PEP692 support
    if tqdm_kwargs is None:
        tqdm_kwargs = {"disable": True}
    jobs = (delayed(func)(**kwargs) for kwargs in func_args)
    # return_as "generator" so that tqdm can monitor progress by consuming the result generator
    result = Parallel(n_jobs=n_jobs, return_as="generator", timeout=timeout)(jobs)  # type: ignore
    result = list(tqdm(result, **tqdm_kwargs))
    return result


###################
# PIPELINE SCRIPT #
###################


def group_normalise_images(
    images: list[Image],
) -> list[Image]:
    images = [i[np.newaxis, ...] if i.ndim == 2 else i for i in images]
    separated_channels = zip(*images)
    combined_channels = (
        [img.flatten() for img in channel] for channel in separated_channels
    )
    combined_channels = [np.concatenate(channel) for channel in combined_channels]
    channel_ranges = [(channel.min(), channel.max()) for channel in combined_channels]
    normalised_images = []
    for img in images:
        norm_img = []
        for channel, in_range in zip(img, channel_ranges):
            # use float32 because float is needed for FFT, and float32 is quicker with little to no practical difference to float64
            channel = exposure.rescale_intensity(
                channel, in_range=in_range, out_range="float32"
            )
            norm_img.append(channel)
        norm_img = np.stack(norm_img, axis=0)
        normalised_images.append(norm_img.squeeze())
    return normalised_images


def image_process_pipeline(
    images: list[Image], atlas: list[Image]
) -> tuple[list[Image], list[shift], list[dict[str, str]], float]:
    # normalise images to the same intensity range float32 (0,1)
    norm_images = group_normalise_images(images)
    # orient images in the same direction
    # tf_images, tfs, shifts, errors = pairwise_image_transform(images)
    tf_images, tfs, shifts, errors = running_mean_image_transform(
        norm_images, 5, channel_axis=None
    )
    # align oriented images to atlas
    consensus_tf, agreement = consensus_align_to_atlas(tf_images, atlas)
    axes_indices = list(range(norm_images[0].ndim))
    # axes_indices.pop(channel_axis) # add this once start supporting channel axes
    for ax_i in axes_indices:
        # if consensus_tf flips the axis, flip the shifts as well
        if ax_i in consensus_tf:
            shifts[:, ax_i] *= -1
    # calculate net transform from image to atlas
    net_tf_to_atlas = [tf + consensus_tf for tf in tfs]
    # atlas-aligned images
    tf_images = [tf(img) for tf, img in zip(net_tf_to_atlas, images)]
    # orientations of the images inferred from atlas
    orientations = [tf.inverse.inferred_orientation() for tf in net_tf_to_atlas]
    return tf_images, shifts, orientations, agreement


def read_image_channel(files: str, key: int) -> Optional[Image]:
    """Read one channel out of a straightened image, or None if it cannot be read.

    A small fraction of the straightened images are written out malformed
    (e.g. a single-channel square image instead of a straightened worm), and
    asking for RAW_STR_CHANNEL raises. Returning None lets the caller skip that
    frame instead of discarding the whole point.
    """
    try:
        return tifffile.imread(files, key=key)
    except Exception:
        return None


def process_point(
    point: int, point_df: pd.DataFrame, atlas_images: list[Image]
) -> pd.DataFrame:
    try:
        args = [
            {"files": file, "key": RAW_STR_CHANNEL}
            for file in point_df[RAW_STR_COL].values
        ]
        point_images = parallel_job(
            read_image_channel,
            args,
            n_jobs=1,
            tqdm_kwargs={
                "desc": f"Loading Point-{point} Images",
                "total": len(args),
                "disable": True,
            },
        )
        # drop unreadable frames rather than losing the whole point to one of them
        readable = [img is not None for img in point_images]
        n_unreadable = len(readable) - sum(readable)
        if n_unreadable:
            print(
                f"Point {point}: skipped {n_unreadable}/{len(readable)} images that could not be read"
            )
        point_df = point_df[readable]
        point_images = list(compress(point_images, readable))
        if not point_images:
            raise ValueError(f"no readable images for point {point}")

        tf_images, shifts, orientations, head_confidence = image_process_pipeline(
            point_images, atlas_images
        )

        ori_df = pd.DataFrame(orientations, index=point_df.index)

        ori_df = point_df[["Time", "Point"]].join(ori_df)
        ori_df["head_confidence"] = head_confidence

        shifts = normalise_shifts(shifts)
        for ax in range(shifts.shape[-1]):
            ori_df[f"shift_ax{ax}"] = shifts[:, ax]

    except Exception:
        print(f"Error processing point {point}")
        print(traceback.format_exc())
        blank_df = pd.DataFrame(columns=["Time", "Point"])
        return blank_df

    # if CREATE_MOVIES:
    #     try:
    #         movie = create_movie(tf_images, shifts)
    #         movie = util.dtype._convert(image=movie, dtype=point_images[0].dtype)
    #         tifffile.imwrite(MOVIE_DIR / f"Point{point:04}_movie.tiff", movie)
    #     except Exception:
    #         print(f"Error creating movie for point {point}")
    #         print(traceback.format_exc())

    return ori_df


def process_df(filemap_df: pd.DataFrame, atlas_images: list[Image]) -> pd.DataFrame:
    t_start = time.time()
    print(
        f"{time.asctime()} - Starting Orientation Prediction for {len(filemap_df)} images..."
    )
    groupby = filemap_df.groupby("Point")
    args = [
        {"point": point, "point_df": point_df, "atlas_images": atlas_images}
        for point, point_df in groupby
    ]
    # no timeout: joblib's timeout aborts the *whole* Parallel call rather than
    # skipping the slow task, so a single long point would throw away every
    # other point's work. The slurm --time limit is the real guard.
    result = parallel_job(
        process_point,
        args,
        n_jobs=THREADS,
        tqdm_kwargs={"desc": "Processing Points", "total": len(args)},
    )
    result = pd.concat(result, axis=0)
    t_end = time.time()
    print(f"{time.asctime()} - Finished Orientation Prediction...")
    mins_duration = int(round((t_end - t_start) / 60))
    print(f"Time taken: {mins_duration}min")
    return result


def load_atlas(csv_file: Path) -> list[Image]:
    atlas_df = pd.read_csv(csv_file)
    args = [
        {"files": file, "key": ATLAS_CHANNEL} for file in atlas_df["atlas_image"].values
    ]
    atlas_images = parallel_job(
        tifffile.imread,
        args,
        tqdm_kwargs={"desc": "Loading Atlas Images", "total": len(args)},
    )
    atlas_images = group_normalise_images(atlas_images)
    return atlas_images


def load_preprocess_filemap(filemap_file: Path) -> pd.DataFrame:
    filemap_df = pd.read_csv(
        filemap_file, usecols=["Time", "Point", RAW_STR_COL, WORM_TYPE_COL]
    )
    filemap_df = filemap_df[filemap_df[WORM_TYPE_COL] == "worm"].drop(
        columns=WORM_TYPE_COL
    )
    filemap_df = filemap_df.sort_values(by=["Point", "Time"])
    return filemap_df


def main():
    global FILEMAP, ATLAS_CSV, OUTPUT_FILE, THREADS

    FILEMAP = Path(FILEMAP)
    if not FILEMAP.exists():
        raise FileNotFoundError(f"Filemap CSV {FILEMAP} does not exist")
    ATLAS_CSV = Path(ATLAS_CSV)
    if not ATLAS_CSV.exists():
        raise FileNotFoundError(f"Atlas CSV {ATLAS_CSV} does not exist")
    OUTPUT_FILE = Path(OUTPUT_FILE)
    # MOVIE_DIR = Path(MOVIE_DIR)

    # CREATE_MOVIES = False

    # if CREATE_MOVIES:
    #     MOVIE_DIR.mkdir(exist_ok=True, parents=True)

    atlas_images = load_atlas(ATLAS_CSV)
    filemap_df = load_preprocess_filemap(FILEMAP)
    result = process_df(filemap_df, atlas_images)
    result.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
