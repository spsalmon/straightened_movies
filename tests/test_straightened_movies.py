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


@pytest.mark.parametrize(
    "reference_shape, moving_shape", [((30, 70), (30, 70)), ((25, 80), (32, 61))]
)
def test_overlap_normalized_cross_correlation_matches_skimage(
    reference_shape, moving_shape
):
    from skimage.registration._masked_phase_cross_correlation import (
        cross_correlate_masked,
    )

    rng = np.random.default_rng(0)
    reference = rng.random(reference_shape)
    moving = rng.random(moving_shape)

    correlation = sm.overlap_normalized_cross_correlation(
        reference, moving, overlap_ratio=0.5
    )
    expected = cross_correlate_masked(
        moving,
        reference,
        np.ones_like(moving, dtype=bool),
        np.ones_like(reference, dtype=bool),
        axes=(0, 1),
        overlap_ratio=0.5,
    )

    # skimage indexes offsets in the opposite direction.
    np.testing.assert_allclose(correlation, expected[::-1, ::-1], atol=1e-8)


def test_estimate_translation_recovers_a_known_shift():
    reference = _blob()
    # The masked correlation only considers shifts leaving 90% of the two masks
    # overlapping, so it is built for jitter rather than for large displacements.
    moving = np.roll(reference, (2, -3), axis=(0, 1))

    shift, dissimilarity = sm.estimate_translation(reference, moving)

    assert tuple(shift) == (-2, 3)
    assert dissimilarity == pytest.approx(0.0, abs=1e-3)


def test_estimate_translation_registers_differently_sized_images():
    texture = np.random.default_rng(0).random((60, 120)).astype(np.float32)
    reference = texture[2:42, 4:105]
    moving = texture[5:50, 9:104]

    shift, dissimilarity = sm.estimate_translation(reference, moving)

    # The origins are 3 and 5 pixels apart; the rest is the difference between
    # the two centres, rounded down.
    assert tuple(shift) == (3 + 22 - 19, 5 + 47 - 50)
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


# ---- Movie assembly ----


def test_orient_images_flips_by_label():
    image = np.array([[1, 2], [3, 4]])
    orientations = [
        {"head": "L", "vulva": "U"},
        {"head": "R", "vulva": "U"},
        {"head": "L", "vulva": "D"},
    ]

    oriented = sm.orient_images([image, image, image], orientations)

    assert np.array_equal(oriented[0], image)
    assert np.array_equal(oriented[1], np.array([[2, 1], [4, 3]]))
    assert np.array_equal(oriented[2], np.array([[3, 4], [1, 2]]))


def _growing_series(lengths, height=10):
    # Frames of growing length, each filled with a constant so the occupied
    # region of the canvas is easy to find.
    return [np.ones((height, length), dtype=np.uint16) for length in lengths]


def test_build_movie_left_keeps_the_head_end_stationary():
    images = _growing_series([10, 20, 30, 40])
    shifts = np.zeros((4, 2), dtype=int)

    movie = sm.build_movie(images, shifts, alignment="left")

    assert movie.shape == (4, 1, 10, 40)
    # Every frame starts at the same column, so growth is entirely rightwards.
    starts = [int(np.flatnonzero(frame[0].any(axis=0))[0]) for frame in movie]
    assert starts == [0, 0, 0, 0]


def test_build_movie_center_spreads_growth_both_ways():
    images = _growing_series([10, 20, 30, 40])
    shifts = np.zeros((4, 2), dtype=int)

    movie = sm.build_movie(images, shifts, alignment="center")

    starts = [int(np.flatnonzero(frame[0].any(axis=0))[0]) for frame in movie]
    ends = [int(np.flatnonzero(frame[0].any(axis=0))[-1]) for frame in movie]
    # The head end moves left as the worm grows, and the tail end moves right.
    assert starts == sorted(starts, reverse=True)
    assert starts[0] > starts[-1]
    assert ends == sorted(ends)


def test_build_movie_preserves_registration_under_both_alignments():
    images = _growing_series([20, 20, 20, 20])
    shifts = np.array([[0, 0], [0, 4], [0, -3], [0, 7]])

    left = sm.build_movie(images, shifts, alignment="left")
    centered = sm.build_movie(images, shifts, alignment="center")

    def starts(movie):
        return [int(np.flatnonzero(frame[0].any(axis=0))[0]) for frame in movie]

    # Alignment shifts every frame by a constant; the spacing between frames,
    # which is what removes the jitter, is identical either way.
    assert np.array_equal(np.diff(starts(left)), np.diff(starts(centered)))
    assert np.array_equal(np.diff(starts(left)), np.diff(shifts[:, 1]))


def test_build_movie_centres_every_frame_dorsoventrally():
    # Straightening puts the midline on the centre row of every frame, so a
    # registered dorsoventral shift can only move the worm off it.
    images = [np.ones((height, 20), dtype=np.uint16) for height in (10, 12, 11, 14)]
    shifts = np.array([[0, 0], [5, 0], [-3, 0], [9, 0]])

    movie = sm.build_movie(images, shifts, alignment="center")

    centres = [
        np.flatnonzero(frame[0].any(axis=1)).mean() for frame in movie
    ]
    assert max(centres) - min(centres) <= 0.5


def test_build_movie_promotes_single_channel_frames():
    movie = sm.build_movie(_growing_series([10, 12]), np.zeros((2, 2), dtype=int))

    assert movie.ndim == 4
    assert movie.shape[1] == 1


def test_build_movie_rejects_an_unknown_alignment():
    with pytest.raises(ValueError, match="alignment must be one of"):
        sm.build_movie(_growing_series([10]), np.zeros((1, 2), dtype=int), "diagonal")


# ---- Block plumbing ----


def test_parse_points_accepts_numbers_and_ranges():
    assert sm.parse_points(["All"]) is None
    assert sm.parse_points(["3", "10-12", "42"]) == {3, 10, 11, 12, 42}


def test_parse_points_rejects_a_malformed_range():
    with pytest.raises(ValueError, match="Could not parse"):
        sm.parse_points(["1-2-3"])


def test_resolve_movie_columns_finds_straightened_columns():
    import polars as pl

    filemap = pl.DataFrame(
        {
            "Time": [0],
            "Point": [1],
            "analysis/ch1_raw_str": ["a.tiff"],
            "analysis/ch2_seg_str": ["b.tiff"],
            "ch2_seg_str_volume": [1.0],
        }
    )

    assert sm.resolve_movie_columns(["All"], filemap) == [
        "analysis/ch1_raw_str",
        "analysis/ch2_seg_str",
    ]
    assert sm.resolve_movie_columns(["analysis/ch1_raw_str"], filemap) == [
        "analysis/ch1_raw_str"
    ]


def test_resolve_movie_columns_rejects_a_missing_column():
    import polars as pl

    filemap = pl.DataFrame({"Time": [0], "Point": [1]})

    with pytest.raises(ValueError, match="not in the filemap"):
        sm.resolve_movie_columns(["analysis/nope_str"], filemap)


def test_read_straightened_image_returns_none_for_an_unreadable_file(tmp_path):
    broken = tmp_path / "broken.tiff"
    broken.write_bytes(b"not a tiff")

    assert sm.read_straightened_image(str(broken), 0) is None


def test_dimension_outliers_flags_isolated_size_jumps_but_not_growth():
    lengths = np.arange(300, 600, 20)
    shapes = np.stack([np.full_like(lengths, 40), lengths], axis=1)
    shapes[4] = (130, shapes[4, 1])
    shapes[9] = (40, shapes[9, 1] // 3)

    outliers = sm.dimension_outliers(shapes)

    assert np.flatnonzero(outliers).tolist() == [4, 9]


def test_read_straightened_image_returns_none_for_a_blank_placeholder(tmp_path):
    import tifffile

    # What a failed straightening leaves behind: one blank plane, whatever channel
    # was asked for.
    placeholder = tmp_path / "placeholder.tiff"
    tifffile.imwrite(placeholder, np.zeros((64, 64), dtype=np.uint8))

    assert sm.read_straightened_image(str(placeholder), 1) is None


def test_read_straightened_image_keeps_a_worm(tmp_path):
    import tifffile

    worm = np.zeros((2, 10, 40), dtype=np.uint16)
    worm[1, 3:7, 5:35] = 100
    path = tmp_path / "worm.tiff"
    tifffile.imwrite(path, worm)

    assert np.array_equal(sm.read_straightened_image(str(path), 1), worm[1])


def test_usable_movie_frames_drops_placeholders_and_odd_frames():
    worm = np.zeros((2, 10, 40), dtype=np.uint16)
    worm[1, 3:7, 5:35] = 100
    # A blank single-plane placeholder, and a two-channel frame of wildly wrong size.
    placeholder = np.zeros((120, 120), dtype=np.uint16)
    oversized = np.ones((2, 100, 40), dtype=np.uint16)
    oversized[1, 0, 0] = 5
    images = [worm] * 6 + [placeholder] + [worm] * 5 + [oversized] + [worm] * 3

    usable = sm.usable_movie_frames(images)

    assert np.flatnonzero(~usable).tolist() == [6, 12]


@pytest.fixture
def synthetic_experiment(tmp_path):
    """A tiny two-point experiment: filemap, straightened images, and an atlas."""
    import polars as pl
    import tifffile

    experiment = tmp_path / "experiment"
    straightened = experiment / "analysis" / "ch1_raw_str"
    report = experiment / "analysis" / "report"
    atlas_dir = tmp_path / "atlas"
    for directory in (straightened, report, atlas_dir):
        directory.mkdir(parents=True)

    rows = []
    for point in (1, 2):
        for time, length in enumerate(range(30, 54, 4)):
            # Two channels; channel 1 carries the orientable structure. Point 2 is
            # imaged facing the other way.
            worm = _worm(length, flip=1 if point == 2 else None)
            image = np.stack([np.zeros_like(worm), worm])
            image = (image * 10_000).astype(np.uint16)
            path = straightened / f"Time{time:06d}_Point{point:06d}_str.tiff"
            tifffile.imwrite(str(path), image, photometric="minisblack")
            rows.append(
                {
                    "Time": time,
                    "Point": point,
                    "analysis/ch1_raw_str": str(path),
                    "ch1_seg_str_qc": "worm",
                }
            )

    filemap_path = report / "analysis_filemap.csv"
    pl.DataFrame(rows).write_csv(str(filemap_path))

    atlas_rows = []
    for length in (30, 40, 50, 60):
        worm = _worm(length)
        image = np.stack([np.zeros_like(worm), worm])
        image = (image * 10_000).astype(np.uint16)
        path = atlas_dir / f"atlas_{length}.tiff"
        tifffile.imwrite(str(path), image, photometric="minisblack")
        atlas_rows.append({"atlas_image": str(path)})

    atlas_csv = atlas_dir / "atlas.csv"
    pl.DataFrame(atlas_rows).write_csv(str(atlas_csv))

    return {
        "experiment": experiment,
        "filemap": filemap_path,
        "atlas": atlas_csv,
        "report": report,
    }


def _run_script(experiment, extra=()):
    import subprocess

    return subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "straightened_movies.py"),
            "--filemap",
            str(experiment["filemap"]),
            "--straightened_column",
            "analysis/ch1_raw_str",
            "--orientation_channel",
            "1",
            "--qc_column",
            "ch1_seg_str_qc",
            "--atlas",
            str(experiment["atlas"]),
            "--atlas_channel",
            "1",
            "--n_jobs",
            "1",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_end_to_end_writes_movies_and_a_cache(synthetic_experiment):
    import polars as pl

    result = _run_script(synthetic_experiment, ["--alignment", "left"])

    assert result.returncode == 0, result.stdout + result.stderr

    movies = synthetic_experiment["experiment"] / "movies" / "ch1_raw_str_movies"
    assert sorted(path.name for path in movies.iterdir()) == [
        "Point0001_movie.tiff",
        "Point0002_movie.tiff",
    ]

    cache = pl.read_csv(str(synthetic_experiment["report"] / "orientations.csv"))
    assert cache.columns == [
        "Time",
        "Point",
        "head",
        "vulva",
        "head_confidence",
        "shift_ax0",
        "shift_ax1",
    ]
    assert cache.height == 12
    # Point 2 was imaged facing the other way, so the two points get opposite labels.
    heads = {
        point: set(cache.filter(pl.col("Point") == point)["head"].to_list())
        for point in (1, 2)
    }
    assert heads[1] != heads[2]


def test_end_to_end_leaves_the_filemap_untouched(synthetic_experiment):
    before = synthetic_experiment["filemap"].read_text()

    _run_script(synthetic_experiment)

    assert synthetic_experiment["filemap"].read_text() == before


def test_end_to_end_reuses_the_cache_on_a_second_run(synthetic_experiment):
    import tifffile

    _run_script(synthetic_experiment, ["--alignment", "center"])
    cache_path = synthetic_experiment["report"] / "orientations.csv"
    cached = cache_path.read_text()

    movies = synthetic_experiment["experiment"] / "movies" / "ch1_raw_str_movies"
    centered = tifffile.imread(str(movies / "Point0001_movie.tiff"))

    result = _run_script(synthetic_experiment, ["--alignment", "left"])
    assert result.returncode == 0, result.stdout + result.stderr

    # The orientations are reused verbatim, but the movie is re-assembled with the
    # new alignment. Which alignment gives the wider canvas depends on whether the
    # registration shifts are jitter or track growth, so only the fact that the
    # movie changed is asserted here; the head-pinning itself is covered by
    # test_build_movie_left_keeps_the_head_end_stationary.
    assert cache_path.read_text() == cached
    assert "Reusing cached orientations" in result.stdout
    assert "Predicting orientations" not in result.stdout
    left = tifffile.imread(str(movies / "Point0001_movie.tiff"))
    assert left.shape != centered.shape or not np.array_equal(left, centered)


def test_end_to_end_reports_a_clear_error_when_no_point_can_be_predicted(
    synthetic_experiment,
):
    # Corrupt every straightened image so that prediction fails for both points.
    for path in (
        synthetic_experiment["experiment"] / "analysis" / "ch1_raw_str"
    ).iterdir():
        path.write_bytes(b"not a tiff")

    result = _run_script(synthetic_experiment)

    assert result.returncode != 0
    assert "No orientations could be predicted" in result.stdout + result.stderr
    # Not a polars schema error leaking out of an empty frame.
    assert "SchemaError" not in result.stdout + result.stderr


def test_end_to_end_skips_placeholders_listed_in_a_stale_cache(synthetic_experiment):
    import polars as pl
    import tifffile

    _run_script(synthetic_experiment)

    # A cache written before placeholders were filtered out still lists them, so
    # replace one cached frame with the blank single plane a failed straightening
    # leaves behind.
    cache = pl.read_csv(str(synthetic_experiment["report"] / "orientations.csv"))
    time = cache.filter(pl.col("Point") == 1)["Time"][2]
    placeholder = (
        synthetic_experiment["experiment"]
        / "analysis"
        / "ch1_raw_str"
        / f"Time{time:06d}_Point000001_str.tiff"
    )
    tifffile.imwrite(str(placeholder), np.zeros((200, 200), dtype=np.uint16))

    result = _run_script(synthetic_experiment)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Error building" not in result.stdout
    movies = synthetic_experiment["experiment"] / "movies" / "ch1_raw_str_movies"
    movie = tifffile.imread(str(movies / "Point0001_movie.tiff"))
    assert movie.shape[0] == cache.filter(pl.col("Point") == 1).height - 1
    assert movie.shape[-2] < 200
