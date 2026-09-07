"""Differentiable PyTorch soft Gaussian splat (CPU + CUDA).

Re-implements a minimal alpha-blended isotropic/anisotropic 2D Gaussian
rasterizer so CF-3DGS-style local photometric SE(3) can run without the
FastGS CUDA extension (``diff_gaussian_rasterization_fastgs``).

Design goals
------------
* Works on CPU-only torch (this box) and on CUDA when available.
* Modest resolutions (e.g. 320–480 width) for interactive CPU solve times.
* Optional ``try_fastgs_rasterizer`` hook that prefers the FastGS CUDA path
  when the extension imports and ``torch.cuda.is_available()``.

Blending
--------
Default mode is a *soft normalized* splat (scatter-add of weighted colors /
weights) — fully vectorized and differentiable, suitable for local SE(3)
photometric solve on CPU. Optional ``mode="sorted"`` does classic front-to-back
alpha compositing (slower; closer to 3DGS).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def quat_normalize(q: Tensor) -> Tensor:
    """Normalize quaternion (…, 4) in (w, x, y, z) convention."""
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def quat_to_rotmat(q: Tensor) -> Tensor:
    """(…, 4) wxyz → (…, 3, 3) rotation matrix."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    B = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return B.reshape(q.shape[:-1] + (3, 3))


def se3_from_quat_trans(q: Tensor, t: Tensor) -> Tensor:
    """Build 4x4 SE(3) from quaternion (wxyz) + translation (3,)."""
    R = quat_to_rotmat(q.reshape(4))
    T = torch.eye(4, device=q.device, dtype=q.dtype)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def transform_points(T: Tensor, pts: Tensor) -> Tensor:
    """Apply 4x4 ``T`` to (N, 3) points → (N, 3)."""
    R = T[:3, :3]
    t = T[:3, 3]
    return pts @ R.T + t


def _ensure_scales3(scales: Tensor) -> Tensor:
    if scales.ndim == 1:
        return scales.unsqueeze(-1).expand(-1, 3)
    if scales.shape[-1] == 1:
        return scales.expand(-1, 3)
    return scales


def project_gaussians(
    means: Tensor,
    scales: Tensor,
    quats: Tensor,
    K: Tensor,
    T_cam: Tensor,
    *,
    isotropic: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project 3D Gaussians into the camera.

    Returns means2d (N,2), depths (N,), cov2d (N,2,2), radii (N,), valid (N,).
    """
    means_c = transform_points(T_cam, means)
    z = means_c[:, 2]
    valid = z > 1e-4

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    inv_z = torch.where(valid, 1.0 / z.clamp_min(1e-4), torch.zeros_like(z))
    u = fx * means_c[:, 0] * inv_z + cx
    v = fy * means_c[:, 1] * inv_z + cy
    means2d = torch.stack([u, v], dim=-1)

    scales = _ensure_scales3(scales).clamp_min(1e-6)

    if isotropic:
        s = scales.mean(dim=-1)
        sx = fx * s * inv_z
        sy = fy * s * inv_z
        cov2d = torch.zeros(means.shape[0], 2, 2, device=means.device, dtype=means.dtype)
        cov2d[:, 0, 0] = sx * sx + 0.3
        cov2d[:, 1, 1] = sy * sy + 0.3
    else:
        R = quat_to_rotmat(quats)
        S = torch.diag_embed(scales)
        cov3d = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        x, y = means_c[:, 0], means_c[:, 1]
        J = torch.zeros(means.shape[0], 2, 3, device=means.device, dtype=means.dtype)
        J[:, 0, 0] = fx * inv_z
        J[:, 0, 2] = -fx * x * inv_z * inv_z
        J[:, 1, 1] = fy * inv_z
        J[:, 1, 2] = -fy * y * inv_z * inv_z
        cov2d = J @ cov3d @ J.transpose(-1, -2)
        eye2 = torch.eye(2, device=means.device, dtype=means.dtype).unsqueeze(0)
        cov2d = cov2d + 0.3 * eye2

    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    ext = torch.sqrt(
        (0.5 * (cov2d[:, 0, 0] - cov2d[:, 1, 1])).pow(2) + cov2d[:, 0, 1].pow(2)
    ).clamp_min(0)
    lam_max = mid + ext
    radii = (3.0 * torch.sqrt(lam_max.clamp_min(1e-8))).clamp(1.0, 24.0)
    return means2d, z, cov2d, radii, valid


def _invert_cov2d(cov2d: Tensor) -> Tensor:
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-8)
    inv = torch.stack(
        [
            torch.stack([c / det, -b / det], dim=-1),
            torch.stack([-b / det, a / det], dim=-1),
        ],
        dim=-2,
    )
    return inv


def _soft_normalized_splat(
    means2d: Tensor,
    depths: Tensor,
    cov2d: Tensor,
    radii: Tensor,
    valid: Tensor,
    colors: Tensor,
    opacities: Tensor,
    height: int,
    width: int,
    bg: Tensor,
) -> Tensor:
    """Vectorized soft splat via scatter-add of local footprints.

    Pixel centers are integer; Mahalanobis distance uses *continuous* means so
    pose / mean gradients flow through the Gaussian falloff. Scatter indices
    are non-diff (STE-style); depth soft-weights provide coarse occlusion.
    """
    device = means2d.device
    dtype = means2d.dtype
    N = means2d.shape[0]
    if N == 0:
        return bg

    inv_cov = _invert_cov2d(cov2d)
    d_med = depths[valid].median() if bool(valid.any()) else torch.tensor(1.0, device=device)
    depth_w = torch.exp(-(depths - d_med).clamp(min=-5.0, max=5.0) / (d_med.abs() + 1e-3))
    depth_w = torch.where(valid, depth_w, torch.zeros_like(depth_w))

    r_int = radii.detach().clamp(1, 8).round().to(torch.long)
    r_max = int(r_int.max().item()) if N > 0 else 1
    offs = torch.arange(-r_max, r_max + 1, device=device)
    oy, ox = torch.meshgrid(offs, offs, indexing="ij")
    ox = ox.reshape(-1)
    oy = oy.reshape(-1)

    uc = means2d[:, 0]
    vc = means2d[:, 1]
    # Integer pixel grid anchored at floor(mean) so neighborhood covers the splat.
    # Gradients flow via dx = pixel - continuous_mean (NOT via the integer anchor).
    u0 = torch.floor(uc).detach()
    v0 = torch.floor(vc).detach()
    xi = (u0.unsqueeze(1) + ox.unsqueeze(0).to(dtype)).long()
    yi = (v0.unsqueeze(1) + oy.unsqueeze(0).to(dtype)).long()
    # Continuous pixel centers (align corners / half-pixel: use integer coords)
    px = xi.to(dtype)
    py = yi.to(dtype)
    inb = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
    rad = r_int.unsqueeze(1).to(dtype)
    in_r = (ox.unsqueeze(0).to(dtype).pow(2) + oy.unsqueeze(0).to(dtype).pow(2)) <= (
        rad * rad + 1e-6
    )
    mask = inb & in_r & valid.unsqueeze(1)

    dx = px - uc.unsqueeze(1)
    dy = py - vc.unsqueeze(1)
    a00 = inv_cov[:, 0, 0].unsqueeze(1)
    a01 = inv_cov[:, 0, 1].unsqueeze(1)
    a11 = inv_cov[:, 1, 1].unsqueeze(1)
    power = -0.5 * (a00 * dx * dx + 2.0 * a01 * dx * dy + a11 * dy * dy)
    gauss = torch.exp(power.clamp(min=-40.0))
    alpha = (opacities.unsqueeze(1) * gauss * depth_w.unsqueeze(1)).clamp(0.0, 1.0)
    alpha = alpha * mask.to(dtype)

    flat_idx = (yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)).reshape(-1)
    w_flat = torch.where(mask.reshape(-1), alpha.reshape(-1), torch.zeros(flat_idx.shape, device=device, dtype=dtype))

    weight_map = torch.zeros(height * width, device=device, dtype=dtype)
    weight_map = weight_map.scatter_add(0, flat_idx, w_flat)

    rgb_acc = torch.zeros(3, height * width, device=device, dtype=dtype)
    for c in range(3):
        cw = (colors[:, c].unsqueeze(1) * alpha).reshape(-1)
        cw = torch.where(mask.reshape(-1), cw, torch.zeros_like(cw))
        rgb_acc[c] = rgb_acc[c].scatter_add(0, flat_idx, cw)

    w = weight_map.view(height, width).clamp_min(0)
    rgb = rgb_acc.view(3, height, width)
    denom = w.unsqueeze(0) + 1e-6
    image = rgb / denom
    cover = (1.0 - torch.exp(-w)).unsqueeze(0)
    image = image * cover + bg * (1.0 - cover)
    return image


def _sorted_alpha_splat(
    means2d: Tensor,
    depths: Tensor,
    cov2d: Tensor,
    radii: Tensor,
    valid: Tensor,
    colors: Tensor,
    opacities: Tensor,
    height: int,
    width: int,
    bg: Tensor,
    chunk: int = 32,
) -> Tensor:
    """Classic front-to-back alpha compositing (slower)."""
    device = means2d.device
    dtype = means2d.dtype
    order = torch.argsort(depths)
    means2d = means2d[order]
    depths = depths[order]
    cov2d = cov2d[order]
    radii = radii[order]
    valid = valid[order]
    opacities = opacities[order]
    colors = colors[order]
    inv_cov = _invert_cov2d(cov2d)

    accum_rgb = torch.zeros(3, height, width, device=device, dtype=dtype)
    accum_a = torch.zeros(height, width, device=device, dtype=dtype)
    N = means2d.shape[0]
    for i in range(N):
        if not bool(valid[i].item()) or float(depths[i]) <= 1e-4:
            continue
        u = means2d[i, 0]
        v = means2d[i, 1]
        r = int(min(8, max(1, float(radii[i]))))
        ui, vi = float(u.item()), float(v.item())
        if ui + r < 0 or vi + r < 0 or ui - r >= width or vi - r >= height:
            continue
        x0 = max(0, int(ui) - r)
        x1 = min(width, int(ui) + r + 1)
        y0 = max(0, int(vi) - r)
        y1 = min(height, int(vi) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys = torch.arange(y0, y1, device=device, dtype=dtype)
        xs = torch.arange(x0, x1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx, dy = xx - u, yy - v
        a00, a01, a11 = inv_cov[i, 0, 0], inv_cov[i, 0, 1], inv_cov[i, 1, 1]
        power = -0.5 * (a00 * dx * dx + 2 * a01 * dx * dy + a11 * dy * dy)
        alpha = (opacities[i] * torch.exp(power.clamp(min=-40.0))).clamp(0.0, 0.99)
        T = (1.0 - accum_a[y0:y1, x0:x1]).clamp_min(0.0)
        weight = alpha * T
        accum_rgb[:, y0:y1, x0:x1] = (
            accum_rgb[:, y0:y1, x0:x1] + weight.unsqueeze(0) * colors[i].view(3, 1, 1)
        )
        accum_a[y0:y1, x0:x1] = accum_a[y0:y1, x0:x1] + weight
    return accum_rgb + (1.0 - accum_a).unsqueeze(0) * bg


def soft_splat(
    means: Tensor,
    colors: Tensor,
    opacities: Tensor,
    scales: Tensor,
    quats: Tensor,
    K: Tensor,
    T_cam: Tensor,
    height: int,
    width: int,
    *,
    bg_color: Union[float, Sequence[float]] = 0.0,
    isotropic: bool = True,
    mode: str = "soft",
) -> Tensor:
    """Project + alpha-blend Gaussians → (3, H, W) image.

    Parameters
    ----------
    mode : ``\"soft\"`` (fast, default) or ``\"sorted\"`` (front-to-back).
    T_cam : 4x4 maps Gaussian means into the camera frame (PoseSequence
        ``T_dst_src`` when means live in the src camera).
    """
    device = means.device
    dtype = means.dtype
    if isinstance(bg_color, (int, float)):
        bg = torch.full((3, height, width), float(bg_color), device=device, dtype=dtype)
    else:
        bg = (
            torch.tensor(list(bg_color), device=device, dtype=dtype)
            .view(3, 1, 1)
            .expand(3, height, width)
            .clone()
        )

    if means.numel() == 0:
        return bg

    means2d, depths, cov2d, radii, valid = project_gaussians(
        means, scales, quats, K, T_cam, isotropic=isotropic
    )
    opacities = opacities.reshape(-1).clamp(0.0, 1.0)
    colors = colors.reshape(-1, 3).clamp(0.0, 1.0)

    if mode == "sorted":
        return _sorted_alpha_splat(
            means2d, depths, cov2d, radii, valid, colors, opacities, height, width, bg
        )
    return _soft_normalized_splat(
        means2d, depths, cov2d, radii, valid, colors, opacities, height, width, bg
    )


def photometric_loss(
    render: Tensor,
    target: Tensor,
    *,
    lambda_dssim: float = 0.2,
) -> Tensor:
    """L1 + (1 - SSIM) matching FastGS / 3DGS training loss."""
    ll1 = (render - target).abs().mean()
    if lambda_dssim <= 0:
        return ll1
    ssim_val = _ssim_simple(render.unsqueeze(0), target.unsqueeze(0))
    return (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim_val)


def _ssim_simple(img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
    """Lightweight SSIM for (1, C, H, W); no fused_ssim dependency."""
    channel = img1.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu12
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    )
    return ssim_map.mean()


def try_fastgs_rasterizer(
    means: Tensor,
    colors: Tensor,
    opacities: Tensor,
    scales: Tensor,
    quats: Tensor,
    K: Tensor,
    T_cam: Tensor,
    height: int,
    width: int,
    *,
    bg_color: Union[float, Sequence[float]] = 0.0,
    isotropic: bool = True,
    **kwargs,
) -> Tensor:
    """Prefer FastGS CUDA rasterizer when available; else ``soft_splat``.

    Stub hook: full FastGS ``GaussianModel`` + viewpoint camera wiring is
    environment-specific. When the extension is missing or CUDA is off, this
    falls back to the pure-PyTorch soft splat (the path used on this box).
    """
    if torch.cuda.is_available():
        try:
            from diff_gaussian_rasterization_fastgs import (  # noqa: F401
                GaussianRasterizationSettings,
                GaussianRasterizer,
            )

            # Extension present — still requires GaussianModel/Camera adapters.
            # Keep soft_splat until those adapters are wired in a CUDA env.
            _ = (GaussianRasterizationSettings, GaussianRasterizer)
        except Exception:
            pass
    return soft_splat(
        means,
        colors,
        opacities,
        scales,
        quats,
        K,
        T_cam,
        height,
        width,
        bg_color=bg_color,
        isotropic=isotropic,
        **kwargs,
    )


__all__ = [
    "quat_normalize",
    "quat_to_rotmat",
    "se3_from_quat_trans",
    "transform_points",
    "project_gaussians",
    "soft_splat",
    "photometric_loss",
    "try_fastgs_rasterizer",
]
