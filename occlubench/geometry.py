"""Vectorised occluder geometry (all envs at once, any backend).

Coverage is estimated by casting rays from the camera to sample points on the task
object's surface and testing them against the occluder rectangle. Each sample is
weighted by its projected image area (cos(theta) / distance^2), so the estimate
approximates the fraction of the object's *pixels* hidden. Self-occlusion by the robot
is ignored (it is the same in clean and perturbed rollouts). The CPU segmentation meter
in ``occlubench.occluder`` is the reference implementation and the tests check agreement.
"""

from __future__ import annotations

import numpy as np

from occlubench.scene import CUBE_HALF


def _cube_samples(k: int = 6):
    g = (np.arange(k) + 0.5) / k * 2 - 1
    u, v = np.meshgrid(g, g)
    u, v = u.ravel(), v.ravel()
    pts, nrm = [], []
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            p = np.zeros((len(u), 3))
            p[:, axis] = sgn
            p[:, (axis + 1) % 3] = u
            p[:, (axis + 2) % 3] = v
            n = np.zeros(3)
            n[axis] = sgn
            pts.append(p * CUBE_HALF)
            nrm.append(np.tile(n, (len(u), 1)))
    return np.concatenate(pts), np.concatenate(nrm)


CUBE_PTS, CUBE_NRM = _cube_samples()


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 4) wxyz, robust branchless-ish implementation."""
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    q = np.empty((len(m), 4))
    tr = m00 + m11 + m22
    w = np.sqrt(np.maximum(0, 1 + tr)) / 2
    x = np.sqrt(np.maximum(0, 1 + m00 - m11 - m22)) / 2
    y = np.sqrt(np.maximum(0, 1 - m00 + m11 - m22)) / 2
    z = np.sqrt(np.maximum(0, 1 - m00 - m11 + m22)) / 2
    q[:, 0] = w
    q[:, 1] = np.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    q[:, 2] = np.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    q[:, 3] = np.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def occluder_frame(cam_pos, cam_mat, target, lateral, depth_frac):
    """Occluder centre and rotation for each env. Plate normal (local z) points along the ray.

    cam_pos (N,3), cam_mat (N,3,3) camera-to-world, target (N,3), lateral (N,).
    Returns pos (N,3), rot (N,3,3), side (N,3) unit lateral direction.
    """
    ray = target - cam_pos
    z = ray / np.linalg.norm(ray, axis=1, keepdims=True)
    right = cam_mat[:, :, 0]
    side = right - z * np.sum(right * z, 1, keepdims=True)
    side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
    y = np.cross(z, side)
    rot = np.stack([side, y, z], axis=2)
    pos = cam_pos + depth_frac[:, None] * ray + lateral[:, None] * side
    return pos, rot, side


def coverage(cam_pos, cam_mat, fovy_deg, aspect, cube_pos, cube_mat, occ_pos, occ_mat, occ_half, active=None):
    """Fraction of the cube's projected area hidden by the occluder, per env (N,). NaN if not in view."""
    pts = cube_pos[:, None] + np.einsum("nij,kj->nki", cube_mat, CUBE_PTS)  # (N,K,3)
    nrm = np.einsum("nij,kj->nki", cube_mat, CUBE_NRM)
    to_cam = cam_pos[:, None] - pts
    dist = np.linalg.norm(to_cam, axis=2)
    cos = np.sum(nrm * to_cam, 2) / dist
    # Frustum test in camera coordinates (camera looks along -z).
    pc = np.einsum("nji,nkj->nki", cam_mat, pts - cam_pos[:, None])
    depth = -pc[..., 2]
    tan_y = np.tan(np.radians(fovy_deg) / 2)
    in_view = (depth > 1e-3) & (np.abs(pc[..., 1]) < tan_y * depth) & (np.abs(pc[..., 0]) < tan_y * aspect * depth)
    w = np.where((cos > 0) & in_view, cos / dist**2, 0.0)
    # Ray-rectangle test in occluder coordinates.
    o = np.einsum("nji,nj->ni", occ_mat, cam_pos - occ_pos)[:, None]  # (N,1,3)
    d = np.einsum("nji,nkj->nki", occ_mat, pts - cam_pos[:, None])  # (N,K,3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[..., 2] / d[..., 2]
    hx = o[..., 0] + t * d[..., 0]
    hy = o[..., 1] + t * d[..., 1]
    hit = (t > 0) & (t < 1) & (np.abs(hx) < occ_half[0]) & (np.abs(hy) < occ_half[1])
    total = w.sum(1)
    hidden = (w * hit).sum(1)
    out = np.where(total > 0, hidden / np.maximum(total, 1e-12), np.nan)
    if active is not None:
        out = np.where(active, out, 0.0)
    return out
