# Canny edge seeding & densification

FastGS normally seeds Gaussians from the COLMAP (or random) point cloud and
densifies from photometric / gradient scores. This fork adds optional **Canny
edge** cues so thin structures get points earlier.

## Flags

| Flag | Group | Default | Meaning |
| --- | --- | --- | --- |
| `--canny_seed` | ModelParams | off | Augment the initial PCD with edge-unprojected points before `create_from_pcd` |
| `--canny_seed_points` | ModelParams | `20000` | Cap on new edge points merged into the PCD |
| `--canny_low` / `--canny_high` | ModelParams | `50` / `150` | OpenCV Canny thresholds |
| `--canny_densify` | OptimizationParams | off | During FastGS densify scoring, also flag Canny edge pixels with high L1 |
| `--canny_edge_loss_thresh` | OptimizationParams | `0.05` | Absolute per-pixel L1 threshold used with the edge mask |
| `--canny_dilate` | OptimizationParams | `1` | Morphological dilation radius (pixels) on the edge mask |

## Seeding (`--canny_seed`)

Implemented in `utils/canny_seed.py` / `utils/canny_utils.py`, wired from
`scene/__init__.py`:

1. Run Canny on each training view.
2. Sample edge pixels (budget split across views).
3. Estimate depth by **IDW** from existing PCD points projected into that view.
4. Unproject to world and concatenate into `BasicPointCloud`.

If IDW cannot find nearby projected supports, that sample is dropped.

## Densification boost (`--canny_densify`)

In `utils/fast_utils.compute_gaussian_score_fastgs`, after the usual
`metric_map = (l1_norm > loss_thresh)`, when the flag is on:

```text
metric_map |= edge_dilated & (l1_abs > canny_edge_loss_thresh)
```

Existing behavior is unchanged when `--canny_densify` is off.

## Example

```bash
python train.py -s /path/to/colmap_scene -m output/run \
  --canny_seed --canny_seed_points 20000 \
  --canny_densify --canny_edge_loss_thresh 0.05
```
