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
from skimage import exposure
from skimage import metrics
from skimage import registration
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
