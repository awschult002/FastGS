"""Multi-keyframe depth-PnP registration for denser pose-graph constraints.

Instead of only adjacent VO (+ fixed skip strides), treat every ``kf_stride``
frames as a keyframe and register frame *i* to the last ``kf_window``
keyframes via depth-PnP. Relatives enter the pose graph by picking the best
match (most inliers among quality-filtered candidates) or by adding all
accepted keyframe edges (``fuse_mode='all'``).

Modes
-----
* ``multi_kf`` — keyframe registration (+ optional ``extra_skip_ks``)
* ``filtered_skips`` — adjacent VO + classic skip-k edges gated by
  ``edge_filter`` (inliers / depth-ratio / reproj / t-scale)

This is complementary to ``depth_pnp_solver`` and does not require photometric
3DGS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .dataset import SequenceDataset
from .depth_pnp_solver import (
    DepthPnPSolveConfig,
    load_depth_npz,
    resize_depth,
    unproject_pixels,
)
from .edge_filter import (
    EdgeQualityConfig,
    evaluate_skip_edge,
    median_depth_ratio,
    median_reproj_error,
)
from .opencv_solver import (
    OpenCVSolveConfig,
    _create_feature_backend,
    _good_matches,
    _load_gray,
    _scaled_K,
)
from .pose_graph import (
    PoseGraphEdge,
    invert_T,
    optimize_pose_graph,
    pose_graph_cost,
)
from .pose_solver import PoseSequence, eye4


@dataclass
class MultiKeyframeConfig:
    """Controls keyframe spacing / fusion on top of DepthPnPSolveConfig."""

    mode: str = "multi_kf"  # "multi_kf" | "filtered_skips"
    kf_stride: int = 5
    kf_window: int = 3
    fuse_mode: str = "best"  # "best" | "all"
    use_edge_filter: bool = True
    include_adjacent: bool = True
    extra_skip_ks: Tuple[int, ...] = ()
    edge_quality: EdgeQualityConfig = field(default_factory=EdgeQualityConfig)
    pose_graph_max_nfev: int = 120
    adjacent_weight: float = 1.0
    # Keep PnP rotation; replace translation with adjacent-chain (SO3 loop closures)
    align_t_to_chain: bool = False


def select_keyframes(n: int, stride: int) -> List[int]:
    stride = max(1, int(stride))
    kfs = list(range(0, n, stride))
    if not kfs or kfs[-1] != n - 1:
        kfs.append(n - 1)
    out: List[int] = []
    seen = set()
    for k in kfs:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


class MultiKeyframeDepthPnPSolver:
    """Adjacent depth-PnP VO + multi-keyframe / filtered-skip edges + pose graph."""

    def __init__(
        self,
        dataset: SequenceDataset,
        depth_paths: List[Optional[str]],
        depth_cfg: Optional[DepthPnPSolveConfig] = None,
        mkf_cfg: Optional[MultiKeyframeConfig] = None,
    ):
        self.dataset = dataset
        self.depth_paths = depth_paths
        self.depth_cfg = depth_cfg or DepthPnPSolveConfig()
        self.mkf_cfg = mkf_cfg or MultiKeyframeConfig()
        if len(depth_paths) != len(dataset):
            raise ValueError(
                f"depth_paths length {len(depth_paths)} != num frames {len(dataset)}"
            )
        self.poses = PoseSequence(num_frames=len(dataset))
        feat_cfg = OpenCVSolveConfig(
            max_features=self.depth_cfg.max_features,
            feature_backend=self.depth_cfg.feature_backend,
        )
        self.detector, self.bf, self.backend_name = _create_feature_backend(feat_cfg)
        self._edge_stats: List[dict] = []

    def solve(self) -> PoseSequence:
        n = len(self.dataset)
        if n < 2:
            raise ValueError("Need >= 2 frames")
        dcfg = self.depth_cfg
        mcfg = self.mkf_cfg
        print(
            f"[multi_kf] solve n={n} mode={mcfg.mode} kf_stride={mcfg.kf_stride} "
            f"kf_window={mcfg.kf_window} fuse={mcfg.fuse_mode} "
            f"edge_filter={mcfg.use_edge_filter} skip_ks={mcfg.extra_skip_ks} "
            f"features={self.backend_name}"
        )

        K_full = self.dataset.intrinsics.as_K()
        scales: List[float] = []
        kps = []
        dess = []
        depths: List[Optional[np.ndarray]] = []

        for i in range(n):
            img, scale = _load_gray(self.dataset.image_paths[i], dcfg.resize_width)
            kp, des = self.detector.detectAndCompute(img, None)
            scales.append(scale)
            kps.append(kp)
            dess.append(des)
            depths.append(self._load_depth_for(i, img.shape[1], img.shape[0]))
            if (i + 1) % 25 == 0 or i == n - 1:
                print(f"[multi_kf] features {i+1}/{n}")

        edges: List[PoseGraphEdge] = []
        self._edge_stats = []
        adj_rels: List[np.ndarray] = []

        for i in range(n - 1):
            K = _scaled_K(K_full, scales[i])
            T_rel, info = self._relative_pose_pnp_rich(
                kps[i], dess[i], kps[i + 1], dess[i + 1], depths[i], depths[i + 1], K
            )
            n_inl = int(info["inliers"])
            if n_inl == 0:
                T_rel = eye4()
            adj_rels.append(T_rel.copy())
            w = mcfg.adjacent_weight * min(1.0, max(n_inl, 1) / 50.0)
            if mcfg.include_adjacent:
                edges.append(PoseGraphEdge(i=i, j=i + 1, T_j_i=T_rel, weight=w))
            self._edge_stats.append(
                {
                    "i": i,
                    "j": i + 1,
                    "kind": "adj",
                    "inliers": n_inl,
                    "weight": w,
                    "t_norm": float(np.linalg.norm(T_rel[:3, 3])),
                }
            )
            print(
                f"[multi_kf] {i:03d}→{i+1:03d} (adj) inliers={n_inl} "
                f"t_norm={np.linalg.norm(T_rel[:3, 3]):.4f}"
            )

        self.poses.set_identity_anchor(0)
        for i, T in enumerate(adj_rels):
            self.poses.set_relative(i, i + 1, T)
        self.poses.compose_forward()
        chain_c2w = [T.copy() for T in self.poses.T_world_cam]

        n_kept = 0
        n_rej = 0

        if mcfg.mode == "multi_kf":
            keyframes = select_keyframes(n, mcfg.kf_stride)
            print(f"[multi_kf] keyframes ({len(keyframes)}): {keyframes[:16]}...")
            for i in range(1, n):
                prior_kfs = [k for k in keyframes if k < i]
                if not prior_kfs:
                    continue
                targets = prior_kfs[-mcfg.kf_window :]
                pair_results: List[Tuple[int, np.ndarray, dict, float]] = []
                for k in targets:
                    if k == i - 1 and mcfg.include_adjacent:
                        continue
                    ok_edge, T_meas, info, weight = self._try_long_edge(
                        k, i, kps, dess, depths, scales, K_full, chain_c2w, mcfg
                    )
                    if not ok_edge:
                        n_rej += 1
                        continue
                    pair_results.append((k, T_meas, info, weight))
                if not pair_results:
                    continue
                if mcfg.fuse_mode == "all":
                    chosen = pair_results
                else:
                    chosen = [max(pair_results, key=lambda x: int(x[2]["inliers"]))]
                for k, T_meas, info, weight in chosen:
                    edges.append(PoseGraphEdge(i=k, j=i, T_j_i=T_meas, weight=weight))
                    n_kept += 1
                    self._edge_stats.append(
                        {
                            "i": k,
                            "j": i,
                            "kind": "mkf",
                            "inliers": int(info["inliers"]),
                            "weight": weight,
                            "depth_ratio": info.get("median_depth_ratio"),
                            "reproj": info.get("median_reproj"),
                        }
                    )
                    if n_kept <= 15 or i % 20 == 0:
                        dr = float(info.get("median_depth_ratio", float("nan")))
                        print(
                            f"[multi_kf] {k:03d}→{i:03d} (mkf) inliers={info['inliers']} "
                            f"w={weight:.3f} dr={dr:.3f}"
                        )

        skip_ks = tuple(mcfg.extra_skip_ks)
        if mcfg.mode == "filtered_skips" and not skip_ks:
            skip_ks = (2, 5, 10)
        for kskip in skip_ks:
            for i0 in range(0, n - int(kskip)):
                j0 = i0 + int(kskip)
                ok_edge, T_meas, info, weight = self._try_long_edge(
                    i0, j0, kps, dess, depths, scales, K_full, chain_c2w, mcfg
                )
                if not ok_edge:
                    n_rej += 1
                    continue
                edges.append(PoseGraphEdge(i=i0, j=j0, T_j_i=T_meas, weight=weight))
                n_kept += 1
                self._edge_stats.append(
                    {
                        "i": i0,
                        "j": j0,
                        "kind": "skip",
                        "inliers": int(info["inliers"]),
                        "weight": weight,
                        "depth_ratio": info.get("median_depth_ratio"),
                        "reproj": info.get("median_reproj"),
                    }
                )

        print(
            f"[multi_kf] long-range edges kept={n_kept} rejected={n_rej} "
            f"total_edges={len(edges)}"
        )

        cost0 = pose_graph_cost(chain_c2w, edges)
        print(f"[multi_kf] pose-graph pre-cost={cost0:.4f}, optimizing...")
        refined = optimize_pose_graph(
            chain_c2w, edges, max_nfev=mcfg.pose_graph_max_nfev
        )
        cost1 = pose_graph_cost(refined, edges)
        print(f"[multi_kf] pose-graph post-cost={cost1:.4f}")
        self.poses.T_world_cam = refined
        for i in range(n - 1):
            T_j_i = invert_T(refined[i + 1]) @ refined[i]
            self.poses.set_relative(i, i + 1, T_j_i)
        return self.poses

    def _try_long_edge(
        self,
        i: int,
        j: int,
        kps,
        dess,
        depths,
        scales,
        K_full,
        chain_c2w,
        mcfg: MultiKeyframeConfig,
    ) -> Tuple[bool, np.ndarray, dict, float]:
        K = _scaled_K(K_full, scales[i])
        T_meas, info = self._relative_pose_pnp_rich(
            kps[i], dess[i], kps[j], dess[j], depths[i], depths[j], K
        )
        T_chain = invert_T(chain_c2w[j]) @ chain_c2w[i]
        if mcfg.use_edge_filter:
            q = evaluate_skip_edge(
                T_meas,
                int(info["inliers"]),
                median_depth_ratio_val=float(
                    info.get("median_depth_ratio", float("nan"))
                ),
                median_reproj_val=float(info.get("median_reproj", float("nan"))),
                T_chain=T_chain,
                cfg=mcfg.edge_quality,
            )
            if not q.accepted:
                return False, T_meas, info, 0.0
            T_out = self._maybe_align_t(T_meas, T_chain, mcfg)
            return True, T_out, info, q.weight
        if int(info["inliers"]) < mcfg.edge_quality.min_inliers:
            return False, T_meas, info, 0.0
        w = mcfg.edge_quality.base_weight * min(1.0, int(info["inliers"]) / 50.0)
        T_out = self._maybe_align_t(T_meas, T_chain, mcfg)
        return True, T_out, info, w

    def _maybe_align_t(self, T_meas, T_chain, mcfg: MultiKeyframeConfig):
        if not mcfg.align_t_to_chain:
            return T_meas
        T = eye4()
        T[:3, :3] = T_meas[:3, :3]
        T[:3, 3] = T_chain[:3, 3]
        return T

    def _load_depth_for(self, idx: int, width: int, height: int) -> Optional[np.ndarray]:
        path = self.depth_paths[idx]
        if path is None:
            return None
        depth = load_depth_npz(path, key=self.depth_cfg.depth_key)
        return resize_depth(depth, width, height)

    def _relative_pose_pnp_rich(
        self,
        kp1,
        des1,
        kp2,
        des2,
        depth1: Optional[np.ndarray],
        depth2: Optional[np.ndarray],
        K: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        info: Dict = {
            "inliers": 0,
            "median_depth_ratio": float("nan"),
            "median_reproj": float("nan"),
        }
        if depth1 is None:
            return eye4(), info
        good = _good_matches(self.bf, des1, des2, self.depth_cfg.match_ratio)
        if len(good) < self.depth_cfg.min_matches:
            return eye4(), info

        pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
        info["median_depth_ratio"] = median_depth_ratio(
            pts1,
            pts2,
            depth1,
            depth2 if depth2 is not None else depth1,
            self.depth_cfg.min_depth,
            self.depth_cfg.max_depth,
        )
        pts3d, valid = unproject_pixels(
            pts1, depth1, K, self.depth_cfg.min_depth, self.depth_cfg.max_depth
        )
        if int(valid.sum()) < self.depth_cfg.min_matches:
            return eye4(), info

        obj = pts3d[valid]
        img_pts = pts2[valid]
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj.astype(np.float64),
                img_pts.astype(np.float64),
                K,
                None,
                iterationsCount=self.depth_cfg.ransac_iters,
                reprojectionError=self.depth_cfg.reproj_err,
                confidence=self.depth_cfg.confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception as exc:
            print(f"[multi_kf] solvePnPRansac failed: {exc}")
            return eye4(), info

        if not ok or rvec is None or tvec is None:
            return eye4(), info

        R, _ = cv2.Rodrigues(rvec)
        T = eye4()
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        inl = (
            np.asarray(inliers).reshape(-1)
            if inliers is not None
            else np.arange(len(obj))
        )
        info["inliers"] = int(len(inl))
        info["median_reproj"] = median_reproj_error(T, obj, img_pts, K, inl)
        return T, info
