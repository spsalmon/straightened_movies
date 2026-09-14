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
