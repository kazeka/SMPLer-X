# Body Measurement Methodology

## Computational pipeline (`measure_bodies.py`)

For each `.npz` detection file:
1. Load the 10 SMPL-X shape `betas` (encodes body proportions in absolute metric units, metres)
2. Run the SMPL-X model in **T-pose** (all joint rotations zeroed out) — canonical shape uncontaminated by the captured pose
3. Get `vertices` (10,475 × 3, metres) and `joints` (145 × 3, metres)
4. Compute measurements from vertices and joints

### Per-measurement method

**Height**

Vertical extent of the mesh (Y-axis is up in SMPL-X convention):
```
vertices[:, 1].max() - vertices[:, 1].min()
```
Spans head top to heel bottom.

**Chest / Waist / Hips — circumferences**

Each is computed as a mesh cross-section perimeter at a horizontal plane:
- Every triangle face is intersected with the plane at height `y`
- Where a triangle edge crosses the plane, the exact crossing point is found by linear interpolation: `t = (y - y_i) / (y_j - y_i)`, `p = v_i + t*(v_j - v_i)`
- Each triangle contributes 0 or 2 crossing points; the segment length is summed
- Total = full perimeter of the cross-section contour at that height

Slice heights are derived from joint positions, not hardcoded:

| Measurement | Slice height |
|-------------|-------------|
| Chest | Midpoint of `Spine_3` and `Neck` joints (upper thorax / pectoral level) |
| Waist | `Spine_2` joint (narrowest trunk section) |
| Hips | Mean Y of `L_Hip` and `R_Hip` joints (widest pelvis section) |

**Shoulder width**

Horizontal (X-axis) distance between left and right shoulder joints:
```
abs(joints[L_SHOULDER, 0] - joints[R_SHOULDER, 0])
```
Joint-to-joint, not a surface measurement.

**Arm span**

Horizontal wrist-to-wrist distance:
```
abs(joints[L_WRIST, 0] - joints[R_WRIST, 0])
```
Valid because T-pose arms are horizontal.

---

## Ground-truth measurement protocol

To compare computed values against real subjects, the tape protocol must match what the model computes.

| Measurement | Protocol |
|-------------|----------|
| **Height** | Subject standing barefoot, heels together, head in Frankfurt plane (eyes horizontal). Measure floor to vertex (top of head). Standard stadiometer reading. |
| **Chest** | Tape around the chest at the **nipple line** (men) or fullest part of the bust (women), arms relaxed at sides. Measure at end of a normal exhale. Corresponds to the mid-Spine_3/Neck slice, roughly the upper-chest / pectoral level. |
| **Waist** | Tape around the **natural waist** — the narrowest visible point of the torso, typically 2–4 cm above the navel. Arms relaxed. Corresponds to Spine_2 (mid-lumbar). |
| **Hips** | Tape around the **maximum circumference of the buttocks**, feet together. Roughly at the level of the greater trochanters, where SMPL-X places the hip joints. |
| **Shoulder width** | Subject standing, arms at sides. Measure the **biacromial width** — straight-line horizontal distance between the two acromion processes (bony tips of the shoulders). Use a sliding calliper or a rigid tape held straight, not following the curve of the shoulder. Matches the joint-to-joint distance the script uses. |
| **Arm span** | Subject standing, arms extended horizontally at shoulder height, palms forward. The script measures **wrist-to-wrist** (not fingertip-to-fingertip), so use a wrist-to-wrist tape with arms in the same horizontal position. |

---

## Known limitations

- **Circumference accuracy**: SMPL-X vertices approximate the skin surface but the mesh under-resolves folds and creases. Expect ±2–4 cm systematic error on circumferences, especially for subjects with higher BMI.
- **Shoulder width**: The script computes joint-to-joint Euclidean distance, which slightly underestimates biacromial breadth because shoulder joints sit below and medial to the acromion. Typical discrepancy is ~1–2 cm.
- **Chest slice level**: SMPL-X Spine_3 and Neck joint heights vary with body proportions. For tall or short-necked subjects the chest slice may land at a slightly different anatomical level than the nipple line.
- **Beta variability across frames**: Each detection is an independent estimate; inter-frame variability in betas is model noise, not real body change. Use the rolling mean from the plot to read the stable per-person estimate.
