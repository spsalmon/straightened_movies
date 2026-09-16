"""Orientation prediction and straightened-movie assembly for one experiment.

For every point, the straightened images classified as worms are aligned to one
another, given an absolute orientation by comparison against a manually oriented
atlas, and assembled into a movie with the frames registered to remove jitter.

Runs standalone or as a ``towbintools_pipeline`` custom building block. It
produces no filemap-visible output: movies are written under ``--movie_dir`` and
the orientation table under ``--orientations``, the latter purely as a cache that
makes re-runs over the same experiment cheap.
"""

import argparse
import os
import pickle
import time
import traceback
from itertools import compress
from itertools import product

import numpy as np
import polars as pl
from joblib import Parallel
from joblib import delayed
from scipy import ndimage
from scipy import signal
from skimage import exposure
from skimage import metrics
from skimage import transform as sk_transform
from tifffile import imwrite
from towbintools.foundation.file_handling import read_filemap
from towbintools.foundation.file_handling import write_filemap
from towbintools.foundation.image_handling import read_tiff_file

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


def dimension_outliers(
    shapes: np.ndarray,
    window: int = 11,
    tolerance: float = 2.0,
) -> np.ndarray:
    """
    Flag frames of a time series whose size departs sharply from their neighbours'.

    Each dimension of each frame is compared against its median over the
    surrounding frames, which follows steady growth while ignoring isolated
    outliers.

    Parameters:
        shapes (np.ndarray): Size of each frame, of shape ``(N, D)``, in time order.
        window (int): Number of frames the median is taken over, centred on each
            frame. (default: 11)
        tolerance (float): Factor by which a dimension may differ from its median,
            in either direction. (default: 2.0)

    Returns:
        np.ndarray: Boolean array of shape ``(N,)``, ``True`` for outlying frames.
    """
    shapes = np.asarray(shapes, dtype=float)
    medians = ndimage.median_filter(shapes, size=(window, 1), mode="nearest")
    ratios = shapes / medians
    return ((ratios > tolerance) | (ratios < 1 / tolerance)).any(axis=1)


def _box_sums(image: np.ndarray, box_shape: tuple[int, ...]) -> np.ndarray:
    # Sum of the image under a box at every offset where the two overlap, as in a
    # "full" correlation, from running sums along one axis at a time.
    for axis, size in enumerate(box_shape):
        length = image.shape[axis]
        padding = [(0, 0)] * image.ndim
        padding[axis] = (1, 0)
        running = np.pad(image.cumsum(axis=axis), padding)
        ends = np.arange(1, length + size)
        image = np.take(running, np.minimum(ends, length), axis=axis) - np.take(
            running, np.maximum(ends - size, 0), axis=axis
        )
    return image


def overlap_normalized_cross_correlation(
    reference: np.ndarray,
    moving: np.ndarray,
    overlap_ratio: float = 0.3,
) -> np.ndarray:
    """
    Normalised cross correlation of two images over their overlap, at every offset.

    Equivalent to scikit-image's masked cross correlation with masks covering both
    images entirely. Every overlap is then a rectangle, so the sums over it come
    from running sums, and a single plain cross correlation is left to compute by
    FFT instead of the six an arbitrary mask needs.

    Parameters:
        reference (np.ndarray): Image of shape ``(H, W)``.
        moving (np.ndarray): Image of shape ``(h, w)``.
        overlap_ratio (float): Offsets overlapping on fewer pixels than this
            fraction of the largest overlap are scored zero. (default: 0.3)

    Returns:
        np.ndarray: The correlation, of shape ``(H + h - 1, W + w - 1)``. Entry
            ``(i, j)`` scores the offset placing pixel ``(0, 0)`` of ``moving`` on
            pixel ``(i - h + 1, j - w + 1)`` of ``reference``.
    """
    reference = reference.astype(np.float64)
    # Flipped, the moving image's overlap sums line up with the reference's.
    flipped = moving[::-1, ::-1].astype(np.float64)

    pixels = np.outer(
        *(
            np.convolve(np.ones(reference_size), np.ones(moving_size))
            for reference_size, moving_size in zip(reference.shape, moving.shape)
        )
    )
    reference_sum = _box_sums(reference, moving.shape)
    moving_sum = _box_sums(flipped, reference.shape)

    # Each term is the (co)variance over the overlap times its pixel count, which
    # cancels out of the ratio.
    covariance = (
        signal.fftconvolve(reference, flipped) - reference_sum * moving_sum / pixels
    )
    reference_variance = (
        _box_sums(reference**2, moving.shape) - reference_sum**2 / pixels
    )
    moving_variance = _box_sums(flipped**2, reference.shape) - moving_sum**2 / pixels
    denominator = np.sqrt(
        np.clip(reference_variance, 0, None) * np.clip(moving_variance, 0, None)
    )

    correlation = np.zeros_like(denominator)
    valid = denominator > 1e3 * np.finfo(denominator.dtype).eps * denominator.max()
    correlation[valid] = np.clip(covariance[valid] / denominator[valid], -1, 1)
    correlation[pixels < overlap_ratio * pixels.max()] = 0
    return correlation


def estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Find the translation that best registers one image onto another.

    Every offset is scored by normalised cross correlation over the overlap of the
    two images alone, so neither edge can wrap onto the other, and the
    dissimilarity is measured over that same overlap once registered.

    Parameters:
        reference (np.ndarray): The image being registered onto, of shape ``(H, W)``.
        moving (np.ndarray): The image being moved, of shape ``(h, w)``.

    Returns:
        tuple[np.ndarray, float]: The integer shift applying ``moving`` onto
            ``reference``, with both images centred on a common canvas, and the
            dissimilarity of the two once registered.
    """
    reference_shape = np.array(reference.shape)
    moving_shape = np.array(moving.shape)
    correlation = overlap_normalized_cross_correlation(
        reference, moving, overlap_ratio=0.9
    )
    # Equal maxima are averaged, as in skimage's masked registration.
    peak = np.argwhere(correlation == correlation.max()).mean(axis=0)
    offset = np.rint(peak - moving_shape + 1).astype(int)

    start = np.maximum(offset, 0)
    stop = np.minimum(reference_shape, moving_shape + offset)
    dissimilarity = structural_dissimilarity(
        reference[start[0] : stop[0], start[1] : stop[1]],
        moving[
            start[0] - offset[0] : stop[0] - offset[0],
            start[1] - offset[1] : stop[1] - offset[1],
        ],
    )

    # Centring rounds down, as padding each image evenly to a common size does.
    centring = (moving_shape - 1) // 2 - (reference_shape - 1) // 2
    return offset + centring, dissimilarity


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
            [
                -value if axis in transforms[-1] else value
                for axis, value in enumerate(shift)
            ]
        )
        transforms.append(transforms[-1] + transform)
        shifts.append(shifts[-1] + shift)

    oriented = [transform(image) for transform, image in zip(transforms, images)]
    return mean_registered_image(oriented, np.array(shifts))


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
        transform: count
        for transform, count in tally.items()
        if length_axis in transform
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

    The frames are placed on a shared canvas by their head-tail registration
    shifts, which removes the jitter between consecutive timepoints, and anchored
    according to ``alignment``. Dorsoventrally they are centred, which keeps the
    midline on one row. Frames must already face head left and vulva up.

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
    # Straightening already puts the midline on the centre row of every frame, so
    # centring is the dorsoventral registration. The measured shift on that axis
    # only adds rounding noise and drift accumulated through the running mean.
    shifts = np.array(shifts, dtype=int)
    shifts[:, 0] = 0
    origins, canvas = registration_origins(
        shapes, shifts, anchors=ALIGNMENTS[alignment]
    )
    return assemble_registered_stack(frames, origins, canvas)


# ---------------------------------------------------------------------------
# Pipeline block
# ---------------------------------------------------------------------------

# Quality-control label marking an image as a usable worm.
DEFAULT_QC_VALUE = "worm"
# Columns of the orientation cache, in order.
# Columns of the orientation cache, with the dtypes they are built with. Stated
# explicitly so that a run where every point failed still produces a typed frame
# rather than an all-null one that fails to join.
ORIENTATION_SCHEMA = {
    "Time": pl.Int64,
    "Point": pl.Int64,
    "head": pl.String,
    "vulva": pl.String,
    "head_confidence": pl.Float64,
    "shift_ax0": pl.Int64,
    "shift_ax1": pl.Int64,
}
ORIENTATION_COLUMNS = list(ORIENTATION_SCHEMA)


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
    if path is None or path == "None":
        return {}
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


def is_blank_image(image: np.ndarray) -> bool:
    """
    Tell whether an image holds a single value throughout.

    When straightening fails, a blank single-plane image the size of the raw field
    of view is written in place of the worm. Such frames can still pass quality
    control, and no real straightened worm is uniform.

    Parameters:
        image (np.ndarray): The image to check.

    Returns:
        bool: ``True`` if every pixel has the same value.
    """
    return image.min() == image.max()


def read_straightened_image(path: str, channel: int) -> np.ndarray | None:
    """
    Read one channel of a straightened image, or ``None`` if it is unusable.

    A small fraction of straightened images are written out malformed or blank.
    Returning ``None`` lets the caller drop that frame instead of losing the whole
    point to it.

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
    if is_blank_image(image):
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
        list[dict]: One record per usable image, with the orientation-cache
            columns. Empty if the point could not be processed.
    """
    try:
        images = [read_straightened_image(path, channel) for path in paths]
        readable = [image is not None for image in images]
        if not all(readable):
            print(
                f"Point {point}: skipped {readable.count(False)}/{len(readable)} "
                "images that were unreadable or blank"
            )
        times = list(compress(times, readable))
        images = list(compress(images, readable))
        if not images:
            raise ValueError(f"No usable images for point {point}")

        normal = ~dimension_outliers([image.shape for image in images])
        if not normal.all():
            print(
                f"Point {point}: skipped {(~normal).sum()}/{len(normal)} "
                "images of abnormal dimensions"
            )
        times = list(compress(times, normal))
        images = list(compress(images, normal))

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


def usable_movie_frames(images: list[np.ndarray]) -> np.ndarray:
    """
    Flag the frames of a point that can go into its movie.

    A blank frame is a placeholder left by a failed straightening. A frame with a
    different number of leading dimensions from most of the others cannot be
    stacked with them, and a frame of outlying size would widen the whole canvas.
    Orientation prediction already skips such frames, but an orientation cache
    written before it did still lists them, so the movie checks again.

    Parameters:
        images (list[np.ndarray]): The point's frames, in time order.

    Returns:
        np.ndarray: Boolean array of shape ``(N,)``, ``True`` for usable frames.
    """
    usable = np.array([not is_blank_image(image) for image in images], dtype=bool)
    leading = [image.shape[:-2] for image in images]
    kept = list(compress(leading, usable))
    if kept:
        common = max(set(kept), key=kept.count)
        usable &= np.array([shape == common for shape in leading], dtype=bool)
    indices = np.flatnonzero(usable)
    if indices.size:
        shapes = [images[index].shape[-2:] for index in indices]
        usable[indices[dimension_outliers(shapes)]] = False
    return usable


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
            usable = usable_movie_frames(images)
            if not usable.all():
                print(
                    f"Point {point}: left {(~usable).sum()}/{len(usable)} blank or "
                    f"malformed frames out of the {column} movie"
                )
            if not usable.any():
                raise ValueError("No usable frames")
            movie = build_movie(
                orient_images(
                    list(compress(images, usable)),
                    list(compress(orientations, usable)),
                ),
                shifts[usable],
                alignment=alignment,
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
    n_jobs = (
        args.n_jobs
        or config.get("n_jobs")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    )
    movie_dir, orientations_path = _resolve_directories(args, config)

    filemap = load_filemap(args.filemap)
    if args.straightened_column not in filemap.columns:
        raise ValueError(
            f"Column {args.straightened_column!r} is not in the filemap. "
            f"Available columns: {filemap.columns}"
        )

    movie_columns = resolve_movie_columns(args.movie_columns, filemap)
    rows = filemap.select(
        list(
            dict.fromkeys(
                ["Time", "Point", args.straightened_column]
                + movie_columns
                + ([args.qc_column] if args.qc_column else [])
            )
        )
    )
    if args.qc_column:
        rows = rows.filter(pl.col(args.qc_column) == args.qc_value)
    selected = parse_points(args.points)
    if selected is not None:
        rows = rows.filter(pl.col("Point").is_in(sorted(selected)))
    rows = rows.sort(["Point", "Time"])
    if rows.height == 0:
        raise ValueError("No images left to process after filtering")

    points = rows["Point"].unique().sort().to_list()
    print(f"{time.asctime()} - Processing {len(points)} points: {points}")

    cached = None
    covered: set[int] = set()
    if not args.recompute_orientations and os.path.exists(orientations_path):
        cached = read_filemap(orientations_path).select(ORIENTATION_COLUMNS)
        covered = set(cached["Point"].unique().to_list())
        print(f"Reusing cached orientations from {orientations_path}")

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
                rows.filter(pl.col("Point") == point)[
                    args.straightened_column
                ].to_list(),
                atlas,
                args.orientation_channel,
            )
            for point in missing
        )
        predicted = pl.DataFrame(
            [record for point_records in records for record in point_records],
            schema=ORIENTATION_SCHEMA,
        )
        if predicted.height == 0 and cached is None:
            raise ValueError(
                "No orientations could be predicted for any of the "
                f"{len(missing)} points. The errors above say why for each one; "
                "unreadable images and a mismatched --orientation_channel are the "
                "usual causes."
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
    print(f"{time.asctime()} - Finished ({round((time.time() - started) / 60)}min)")


if __name__ == "__main__":
    main()
