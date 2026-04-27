"""
Compute average body measurements (mm) from SMPLer-X inference results.

No camera calibration required: SMPL-X shape betas encode body dimensions in
absolute metric units. We run the model in T-pose (all joints zeroed) so
measurements reflect body shape alone, independent of the captured pose or
camera parameters. The saved `transl` values are intentionally ignored.

Usage:
    python measure_bodies.py demo/results/myvideo/
    python measure_bodies.py demo/results/myvideo/ --model_path common/utils/human_model_files
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# SMPL-X orig_joint_part body joint indices (from common/utils/human_models.py):
#   0  Pelvis      1  L_Hip       2  R_Hip
#   3  Spine_1     4  L_Knee      5  R_Knee
#   6  Spine_2     7  L_Ankle     8  R_Ankle
#   9  Spine_3    10  L_Foot     11  R_Foot
#  12  Neck       13  L_Collar   14  R_Collar
#  15  Head       16  L_Shoulder 17  R_Shoulder
#  18  L_Elbow    19  R_Elbow    20  L_Wrist    21  R_Wrist
PELVIS      = 0
L_HIP, R_HIP = 1, 2
SPINE_2     = 6
SPINE_3     = 9
NECK        = 12
L_SHOULDER, R_SHOULDER = 16, 17
L_WRIST, R_WRIST       = 20, 21


def load_smplx(model_path: str):
    try:
        import smplx
    except ImportError:
        sys.exit("smplx not installed. Run: pip install smplx")

    # Use SMPLX directly rather than smplx.create(), which infers model type
    # from the path string and misidentifies 'human_model_files' as 'human'.
    smplx_dir = os.path.join(model_path, "smplx")
    return smplx.SMPLX(
        smplx_dir,
        gender="neutral",
        num_betas=10,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    )


def tpose_mesh(model, betas: np.ndarray):
    """Forward pass with zero pose, returning (vertices, joints) in metres."""
    b = torch.tensor(betas.reshape(1, 10), dtype=torch.float32)
    with torch.no_grad():
        out = model(betas=b, return_verts=True)
    return out.vertices.squeeze().numpy(), out.joints.squeeze().numpy()


def cross_section_perimeter(vertices: np.ndarray, faces: np.ndarray, y: float) -> float | None:
    """
    Exact cross-section perimeter at height `y` by intersecting every
    mesh triangle with the horizontal plane and summing edge lengths.
    Returns metres, or None if the plane misses the mesh.
    """
    total = 0.0
    count = 0
    for tri_idx in faces:
        tri = vertices[tri_idx]          # (3, 3)
        above = tri[:, 1] > y

        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if above[i] != above[j]:
                t = (y - tri[i, 1]) / (tri[j, 1] - tri[i, 1])
                pts.append(tri[i] + t * (tri[j] - tri[i]))

        if len(pts) == 2:
            total += np.linalg.norm(pts[1] - pts[0])
            count += 1

    return total if count > 0 else None


def measure(vertices: np.ndarray, joints: np.ndarray, faces: np.ndarray) -> dict:
    """
    Returns measurements in metres.

    Anatomical slice heights:
      chest  → midpoint between Spine_3 and Neck  (upper-torso / pectoral level)
      waist  → Spine_2                             (narrowest trunk section)
      hips   → mean of L_Hip / R_Hip joints        (widest lower-body section)
    """
    height = float(vertices[:, 1].max() - vertices[:, 1].min())

    y_chest = float((joints[SPINE_3, 1] + joints[NECK, 1]) / 2)
    y_waist = float(joints[SPINE_2, 1])
    y_hips  = float((joints[L_HIP, 1] + joints[R_HIP, 1]) / 2)

    # Shoulder width: horizontal distance between shoulder joints
    shoulder_width = float(abs(joints[L_SHOULDER, 0] - joints[R_SHOULDER, 0]))

    # Arm span: wrist-to-wrist (T-pose arms are horizontal)
    arm_span = float(abs(joints[L_WRIST, 0] - joints[R_WRIST, 0]))

    return {
        "height":         height,
        "chest":          cross_section_perimeter(vertices, faces, y_chest),
        "waist":          cross_section_perimeter(vertices, faces, y_waist),
        "hips":           cross_section_perimeter(vertices, faces, y_hips),
        "shoulder_width": shoulder_width,
        "arm_span":       arm_span,
    }


def load_betas(npz_path: str) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    return data["betas"].reshape(10).astype(np.float32)


def frame_index(npz_path: str) -> int:
    """Extract frame number from filename, e.g. '00042_1.npz' → 42."""
    return int(os.path.basename(npz_path).split("_")[0])


def plot_measurements(
    frame_indices: list[int],
    per_detection: dict[str, list[float]],
    save_path: str,
    window: int = 30,
) -> None:
    keys = list(per_detection.keys())
    n = len(keys)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3), sharex=True)
    axes = axes.flatten()

    frames = np.array(frame_indices)

    for ax, key in zip(axes, keys):
        vals_mm = np.array(per_detection[key]) * 1000
        ax.scatter(frames, vals_mm, s=4, alpha=0.35, color="steelblue", label="per detection")

        # rolling mean (sorted by frame index)
        order = np.argsort(frames)
        f_sorted = frames[order]
        v_sorted = vals_mm[order]
        if len(v_sorted) >= window:
            kernel = np.ones(window) / window
            rolling = np.convolve(v_sorted, kernel, mode="valid")
            ax.plot(f_sorted[window - 1:], rolling, color="crimson", linewidth=1.5, label=f"rolling mean (w={window})")

        ax.set_title(key.replace("_", " ").title())
        ax.set_ylabel("mm")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    for ax in axes[n:]:
        ax.set_visible(False)

    for ax in axes[(nrows - 1) * ncols: n]:
        ax.set_xlabel("frame index")

    fig.suptitle("Body measurements vs frame index", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def print_stats(label: str, values_m: list[float]) -> None:
    if not values_m:
        print(f"  {label:<18}: no data")
        return
    arr = np.array(values_m) * 1000  # → mm
    print(f"  {label:<18}: {arr.mean():.1f} mm  ±{arr.std():.1f}  "
          f"[{arr.min():.1f} – {arr.max():.1f}]  n={len(arr)}")


def main():
    parser = argparse.ArgumentParser(description="Body measurement averages from SMPLer-X results")
    parser.add_argument("results_dir", help="Path to demo/results/{VIDEO_NAME}/")
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _default_model_path = os.path.join(_repo_root, "common", "utils", "human_model_files")
    parser.add_argument("--model_path", default=_default_model_path,
                        help="Path to SMPL-X model files")
    parser.add_argument("--mean_betas", action="store_true",
                        help="Also show measurements for the mean betas across all detections")
    parser.add_argument("--no-plot", dest="no_plot", action="store_true",
                        help="Skip saving the measurements plot")
    parser.add_argument("--rolling-window", type=int, default=30, metavar="N",
                        help="Rolling mean window size for the plot (default: 30)")
    args = parser.parse_args()

    smplx_dir = os.path.join(args.results_dir, "smplx")
    npz_files = sorted(glob.glob(os.path.join(smplx_dir, "*.npz")))
    if not npz_files:
        sys.exit(f"No .npz files found under {smplx_dir}")

    print(f"Loading {len(npz_files)} detections from {smplx_dir}")

    model = load_smplx(args.model_path)
    model.eval()
    faces = model.faces  # (N_faces, 3) int array

    all_betas = []
    frame_indices: list[int] = []
    per_detection: dict[str, list[float]] = {
        k: [] for k in ("height", "chest", "waist", "hips", "shoulder_width", "arm_span")
    }

    for path in tqdm(npz_files, desc="Processing detections", unit="det"):
        betas = load_betas(path)
        all_betas.append(betas)
        frame_indices.append(frame_index(path))
        verts, joints = tpose_mesh(model, betas)
        m = measure(verts, joints, faces)
        for k, v in m.items():
            if v is not None:
                per_detection[k].append(v)

    print("\n=== Per-detection averages ===")
    for key in per_detection:
        print_stats(key, per_detection[key])

    if not args.no_plot:
        plot_path = os.path.join(args.results_dir, "measurements.png")
        plot_measurements(frame_indices, per_detection, plot_path, window=args.rolling_window)

    if args.mean_betas:
        mean_betas = np.mean(all_betas, axis=0)
        verts, joints = tpose_mesh(model, mean_betas)
        m = measure(verts, joints, faces)
        print("\n=== Mean-betas body (single representative person) ===")
        for key, val in m.items():
            if val is not None:
                print(f"  {key:<18}: {val * 1000:.1f} mm")


if __name__ == "__main__":
    main()
