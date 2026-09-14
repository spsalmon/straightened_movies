"""Orientation prediction and straightened-movie assembly for one experiment.

For every point, the straightened images classified as worms are aligned to one
another, given an absolute orientation by comparison against a manually oriented
atlas, and assembled into a movie with the frames registered to remove jitter.

Runs standalone or as a ``towbintools_pipeline`` custom building block. It
produces no filemap-visible output: movies are written under ``--movie_dir`` and
the orientation table under ``--orientations``, the latter purely as a cache that
makes re-runs over the same experiment cheap.
"""

from itertools import compress
from itertools import product

import numpy as np
from skimage import exposure
from skimage import metrics
from skimage import registration
from skimage import transform as sk_transform
from towbintools.foundation.image_handling import pad_to_dim_equally

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
        return isinstance(other, FlipTransform) and self.mirror_axes == other.mirror_axes

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
        reference = mean_registered_image(oriented[-window:], np.array(shifts[-window:]))
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
