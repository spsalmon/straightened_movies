# Translation scoring: which roll is right?

Date: 2026-09-14

## The question

`estimate_translation` registers a moving image onto a reference, then measures how
dissimilar the two are once registered. That dissimilarity is what ranks the four
candidate flips in `estimate_flip_transform`, so it decides which way a worm is
judged to face.

The original implementation (`legacy/ori_align.py:232`) rolled the moving image
into place with

```python
mov_padded = np.roll(mov_padded, shift)
```

With no `axis` argument, numpy flattens the array, rolls it by `dy + dx`, and
restores the shape. That is not a translation: the vertical component is folded
into the horizontal one and rows wrap into each other. The dissimilarity was
therefore measured on an image that had not actually been registered.

Every other call site in the old code passes an axis (`create_movie` uses
`axis=(-2, -1)`, `_mean_image` uses `axis=(0, 1)`), so this looks like an
oversight rather than a deliberate choice.

The estimated *shift* is unaffected — it comes straight out of
`phase_cross_correlation`. Only the score changes. Checked directly:

```
new : (array([-5,  1]), 0.2420)
old : (array([-5,  1]), 0.5644)
```

Same shift, different dissimilarity.

## The measurement

Five points of

```
/mnt/towbin.data/shared/kstojanovski/20240202_Orca_10x_yap-1del_col-10-tir_wBT160-186-310-337-380-393_25C_20240202_171239_051
```

straightened column `analysis_sacha/ch1_ch2_raw_str`, orientation channel 1, QC
column `ch2_seg_str_worm_type`, atlas
`/mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv` channel 1.
Between 319 and 379 images per point.

The two arms differ **only** in the roll inside `estimate_translation`;
everything else, including the new frame-placement geometry, is held constant.
`head_confidence` — the share of atlas comparisons agreeing on the head direction
— is the score.

| Point | Images | Per-axis roll | Flattened roll |
|---|---|---|---|
| 1 | 319 | **0.8519** | 0.7222 |
| 2 | 320 | **0.8333** | 0.7407 |
| 3 | 374 | **0.9259** | 0.6667 |
| 4 | 379 | **0.9273** | 0.9091 |
| 5 | 345 | **0.8621** | 0.8103 |
| **mean** | | **0.8801** | 0.7698 |

Roughly 16 minutes per arm.

## Conclusion

The per-axis roll wins on every point, by 11 percentage points on average and by
26 on the worst case (point 3: 0.926 against 0.667). **It ships**, which is what
`straightened_movies.py` already does.

This is a better-scored search, not merely a tidier one: measuring dissimilarity
on a properly registered pair separates the four candidate flips more sharply, so
more of the per-frame atlas comparisons land on the same answer.

## Caveat on comparing against old results

`legacy/example_output.csv`, produced by the original script end to end, records
0.788, 0.903, 1.000, 0.720 and 0.746 for these five points — which matches
neither arm. That is expected: it also predates the frame-placement change, whose
centred mean image no longer wraps content around the canvas edge. The table
above isolates the roll alone and is the comparison the decision rests on.

Orientation labels themselves were separately checked against the legacy
implementation on synthetic series and match exactly for unflipped, head-flipped
and vulva-flipped input.
