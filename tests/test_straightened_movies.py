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
    assert (origins - anchored == np.array(canvas) // 2).all()
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


# ---- Normalisation, similarity, translation ----


def test_normalize_images_to_common_range_uses_one_range_for_the_series():
    images = [
        np.array([[0, 50]], dtype=np.uint16),
        np.array([[50, 100]], dtype=np.uint16),
    ]

    normalized = sm.normalize_images_to_common_range(images)

    # A shared range means the 50 in both images maps to the same value, which a
    # per-image normalisation would not do.
    assert normalized[0][1] == pytest.approx(normalized[1][0])
    assert normalized[0][0] == pytest.approx(0.0)
    assert normalized[1][1] == pytest.approx(1.0)


def test_normalize_images_to_common_range_keeps_dimensionality():
    two_d = [np.ones((4, 5), dtype=np.uint16), np.zeros((4, 5), dtype=np.uint16)]
    three_d = [
        np.ones((2, 4, 5), dtype=np.uint16),
        np.zeros((2, 4, 5), dtype=np.uint16),
    ]

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

    assert sm.structural_dissimilarity(
        image, slightly_off
    ) < sm.structural_dissimilarity(image, very_off)


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
    # The masked correlation only considers shifts leaving 90% of the two masks
    # overlapping, so it is built for jitter rather than for large displacements.
    moving = np.roll(reference, (2, -3), axis=(0, 1))

    shift, dissimilarity = sm.estimate_translation(reference, moving)

    assert tuple(shift) == (-2, 3)
    assert dissimilarity == pytest.approx(0.0, abs=1e-3)


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


def test_estimate_flip_transform_recovers_a_flip():
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
    sides = [
        image[:, : image.shape[1] // 2].sum() > image[:, image.shape[1] // 2 :].sum()
        for image in oriented
    ]
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
