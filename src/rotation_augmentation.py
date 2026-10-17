"""Synthetic in-plane camera rotation for CDnet video sequences.

Drop this file into src/rotation_augmentation.py in your project.

- Angles refer to the original, unrotated reference sequence.
- Each sequence starts at angle zero (frame 0 is unrotated).
- OpenCV's positive angles are counterclockwise in the displayed image.
- The same affine transform is applied to each frame and optional CDnet mask.
- Returned forward/inverse matrices also let you transform object coordinates.
"""

from __future__ import annotations

from hashlib import blake2b
from typing import Sequence

import cv2
import numpy as np


def stable_seed(sequence_id: str, base_seed: int = 42) -> int:
    """Reproducible per-sequence seed (unlike Python's randomized hash())."""
    key = f"{base_seed}:{sequence_id}".encode("utf-8")
    return int.from_bytes(blake2b(key, digest_size=4).digest(), "little")


def sample_rotation_angles(
    num_frames: int,
    *,
    max_angle_deg: float = 15.0,
    max_step_deg: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Sample a smooth, bounded camera-angle trajectory.

    Returns shape (T,) in degrees, with angles[0] == 0.
    ``max_angle_deg`` bounds orientation relative to the first frame.
    ``max_step_deg`` bounds rotation between neighboring frames.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    if max_angle_deg <= 0 or max_step_deg <= 0:
        raise ValueError("max_angle_deg and max_step_deg must be positive")

    rng = np.random.default_rng(seed)
    angles = np.zeros(num_frames, dtype=np.float32)
    angular_velocity = float(rng.uniform(-max_step_deg, max_step_deg))

    for i in range(1, num_frames):
        # Smoothly varying angular velocity, rather than random independent angles.
        perturbation = rng.normal(0.0, max_step_deg)
        angular_velocity = float(
            np.clip(0.85 * angular_velocity + 0.15 * perturbation,
                    -max_step_deg, max_step_deg)
        )
        proposal = float(angles[i - 1] + angular_velocity)
        if abs(proposal) > max_angle_deg:
            # Gently reverse at the allowed range instead of going out of bounds.
            angular_velocity *= -0.8
            proposal = float(angles[i - 1] + angular_velocity)
        angles[i] = np.clip(proposal, -max_angle_deg, max_angle_deg)

    return angles


def rotate_sequence(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    masks: Sequence[np.ndarray] | np.ndarray | None = None,
    angles_deg: Sequence[float] | np.ndarray | None = None,
    max_angle_deg: float = 15.0,
    max_step_deg: float = 2.0,
    seed: int | None = None,
) -> dict:
    """Rotate frames and optionally matching CDnet ground-truth masks.

    Args:
        frames: Sequence of images, shape (T,H,W) or (T,H,W,C).
        masks: Optional CDnet label images, shape (T,H,W).
        angles_deg: Optional explicit rotation labels, length T (e.g. fixed test angles).
        max_angle_deg, max_step_deg, seed: Used when angles_deg isn't supplied.

    Returns a dictionary containing:
        frames: rotated images (same shape and dtype as input)
        masks: rotated labels (or None); newly exposed pixels are 85 (invalid ROI)
        angles_deg: applied orientation per frame, shape (T,)
        delta_angles_deg: relative rotation from preceding frame, starting with 0
        matrices: original -> rotated, shape (T,2,3)
        inverse_matrices: rotated -> original, shape (T,2,3)
        valid_masks: which output pixels originate in the input image, shape (T,H,W)
    """
    frames = np.asarray(frames)
    if frames.ndim not in (3, 4):
        raise ValueError("frames must have shape (T,H,W) or (T,H,W,C)")
    t, h, w = frames.shape[:3]
    if t < 1:
        raise ValueError("frames must contain at least one image")

    if masks is not None:
        masks = np.asarray(masks)
        if masks.shape != (t, h, w):
            raise ValueError(f"masks must have shape {(t, h, w)}, got {masks.shape}")

    if angles_deg is None:
        angles = sample_rotation_angles(
            t, max_angle_deg=max_angle_deg,
            max_step_deg=max_step_deg, seed=seed,
        )
    else:
        angles = np.asarray(angles_deg, dtype=np.float32)
        if angles.shape != (t,) or not np.all(np.isfinite(angles)):
            raise ValueError(f"angles_deg must contain {t} finite angle values")

    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    rotated_frames = []
    rotated_masks = [] if masks is not None else None
    valid_masks = []
    matrices = []
    inverse_matrices = []

    for i, angle in enumerate(angles):
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        M_inv = cv2.invertAffineTransform(M)
        matrices.append(M.astype(np.float32))
        inverse_matrices.append(M_inv.astype(np.float32))

        # Reflected borders avoid large black corners, but see note in README:
        # a model can still exploit border artifacts, so compare with valid_masks.
        rotated_frames.append(
            cv2.warpAffine(
                frames[i], M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        )
        valid_masks.append(
            cv2.warpAffine(
                np.ones((h, w), dtype=np.uint8), M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
        )
        if masks is not None:
            rotated_masks.append(
                cv2.warpAffine(
                    masks[i], M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=85,  # CDnet: invalid / outside ROI
                )
            )

    return {
        "frames": np.stack(rotated_frames),
        "masks": np.stack(rotated_masks) if masks is not None else None,
        "angles_deg": angles.copy(),
        "delta_angles_deg": np.diff(angles, prepend=angles[0]),
        "matrices": np.stack(matrices),
        "inverse_matrices": np.stack(inverse_matrices),
        "valid_masks": np.stack(valid_masks),
    }


def transform_points(points_xy: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """Apply a transform per frame to points (T,2) or (T,K,2).

    Use output['matrices'] for original -> rotated and
    output['inverse_matrices'] for rotated -> original.
    Points are (x,y), *not* NumPy's (row,column) indexing.
    """
    points = np.asarray(points_xy, dtype=np.float32)
    matrices = np.asarray(matrices, dtype=np.float32)
    if (points.ndim not in (2, 3) or points.shape[-1] != 2 or
            matrices.shape != (len(points), 2, 3)):
        raise ValueError("points must be (T,2) or (T,K,2); matrices must be (T,2,3)")
    linear = matrices[:, :, :2]
    translation = matrices[:, :, 2]
    result = np.einsum("tij,t...j->t...i", linear, points)
    if points.ndim == 3:
        translation = translation[:, None, :]
    return result + translation
