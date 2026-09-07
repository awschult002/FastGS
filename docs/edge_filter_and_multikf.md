# Skip-edge quality filter & multi-keyframe registration

## Problem
Pose-graph skip edges from mono-depth PnP (especially `k=10`) often have
translation scales ~0.07× the adjacent chain. Unfiltered skips **hurt** ATE
(Francis-40: baseline ~0.036 → naive PG ~0.113).

## `cf3dgs_bridge/edge_filter.py`
Accept a skip/long-range edge only if:
- inliers ≥ `min_inliers` (default 30)
- median depth ratio `depth_j/depth_i` ∈ [0.5, 2]
- median reprojection error ≤ 3 px
- optional `||t_meas||/||t_chain||` ∈ [0.5, 2]
- optional rotation disagreement vs chain ≤ 0.25 rad

```python
from cf3dgs_bridge.edge_filter import EdgeQualityConfig, evaluate_skip_edge
q = evaluate_skip_edge(T_meas, n_inliers, median_depth_ratio_val=dr,
                       median_reproj_val=rp, T_chain=T_chain)
if q.accepted:
    edges.append(PoseGraphEdge(i=i, j=j, T_j_i=T_meas, weight=q.weight))
```

## `cf3dgs_bridge/multi_keyframe.py`
- `mode=multi_kf`: every `kf_stride` frames is a keyframe; register frame i to
  the last `kf_window` keyframes; fuse with `best` (max inliers) or `all`.
- `mode=filtered_skips`: adjacent VO + classic skip-ks gated by edge_filter.

## Eval
```bash
.venv/bin/python scripts/eval_accuracy_channels.py \
  --scene /path/to/Francis --max_frames 40 --channel filtered_skips \
  --out_json eval_out/francis_filtered_skips_40.json
```
