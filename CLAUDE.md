# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMPLer-X is a research codebase for expressive human pose and shape estimation (EHPS) using SMPL-X body models. It estimates whole-body parameters (body, hands, face) from single images using a ViT-based encoder with task-token-driven regressors. All scripts run via Slurm (`srun`) on a cluster named `Zoetrope`.

## Environment Setup

```bash
conda create -n smplerx python=3.8 -y
conda activate smplerx
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install -r requirements.txt

# install mmpose (required for the ViTPose encoder)
cd main/transformer_utils
pip install -v -e .
cd ../..
```

**Known issues:**
- If you get `RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported`, patch `torchgeometry` per [this fix](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).
- If you get `KeyError: 'SinePositionalEncoding is already registered'`, add `force=True` to the `@POSITIONAL_ENCODING.register_module()` decorator in `main/transformer_utils/mmpose/models/utils/positional_encoding.py`.

## Training

```bash
cd main
sh slurm_train.sh {JOB_NAME} {NUM_GPU} {CONFIG_FILE}

# Example: train SMPLer-X-H32 with 16 GPUs
sh slurm_train.sh smpler_x_h32 16 config_smpler_x_h32.py
```

- Config files live in `main/config/`. The naming convention is `config_smpler_x_{size}{num_datasets}.py` (e.g., `config_smpler_x_h32.py` = ViT-Huge, 32 datasets).
- Fine-tuning configs for specific datasets are in `main/config/config_ft_*.py`.
- Training output (checkpoints, logs, a copy of the config) is saved to `output/train_{JOB_NAME}_{DATETIME}/`.
- Checkpoints are saved as `model_dump/snapshot_{EPOCH}.pth.tar`. The SMPL-X layer weights are intentionally excluded from saves.

## Testing

```bash
cd main
sh slurm_test.sh {JOB_NAME} {NUM_GPU} {TRAIN_OUTPUT_DIR} {CKPT_ID}

# Example: evaluate epoch 9 from a training run
sh slurm_test.sh eval_h32 1 train_smpler_x_h32_20231001_120000 9
```

- The test script reads the config from `output/{TRAIN_OUTPUT_DIR}/code/config_base.py` (auto-copied during training).
- `NUM_GPU=1` is recommended for testing.
- Test results are saved to `output/test_{JOB_NAME}_ep{CKPT_ID}_{TESTSET}/`.
- `slurm_test.sh` runs evaluation across multiple benchmarks sequentially (PW3D, EgoBody, UBody, EHF, AGORA, ARCTIC, RenBody).

## Inference on Video

```bash
cd main
sh slurm_inference.sh {VIDEO_FILE_BASENAME} {FORMAT} {FPS} {PRETRAINED_CKPT}

# Example
sh slurm_inference.sh test_video mp4 24 smpler_x_h32
```

- Place input video at `demo/videos/{VIDEO_FILE}.{FORMAT}`.
- Pretrained models must be in `pretrained_models/` (download from HuggingFace links in README).
- Inference also requires an mmdet Faster-RCNN model at `pretrained_models/mmdet/`.
- Output (per-frame SMPL-X params as `.npz`, meshes as `.obj`, rendered images) is saved to `demo/results/{VIDEO_FILE}/`.

### Mesh Overlay Visualization

```bash
ffmpeg -i {VIDEO_FILE} -f image2 -vf fps=30 {INFERENCE_DIR}/{VIDEO_NAME}/orig_img/%06d.jpg -hide_banner -loglevel error

cd main && python render.py \
    --data_path {INFERENCE_DIR} --seq {VIDEO_NAME} \
    --image_path {INFERENCE_DIR}/{VIDEO_NAME} \
    --render_biggest_person False
```

## Architecture

The forward pass in `main/SMPLer_X.py:Model.forward()` has a clear pipeline:

1. **Encoder** (`main/transformer_utils/` — a modified ViTPose): Takes a body crop and outputs image features + task tokens. Task tokens are split into: shape, camera, expression, jaw pose, hand (×2), and body pose tokens.
2. **Body regressor** (`common/nets/smpler_x.py:BodyRotationNet`, `PositionNet`): Predicts body joint heatmaps and SMPL-X body pose (root + 21 joints in rot6d), shape, and camera from body task tokens.
3. **Box net** (`BoxNet`): Predicts bounding boxes for left hand, right hand, and face from body image features.
4. **Hand RoI + regressor** (`HandRoI`, `HandRotationNet`): Crops and upsamples hand features, then predicts hand joint positions and poses. Left hand is flipped to right-hand space and un-flipped after inference.
5. **Face regressor** (`FaceRegressor`): Predicts expression and jaw pose from face task tokens.
6. **SMPL-X layer**: Assembles all parameters (root, body, lhand, rhand, jaw poses + shape + expr) into a 3D mesh via `smpl_x.layer['neutral']`.

All pose outputs are in rot6d format; they are converted to axis-angle via `utils.transforms.rot6d_to_axis_angle` before passing to the SMPL-X layer.

## Configuration System

Config files are Python files parsed by `mmcv.Config`. `main/config.py:Config` loads them and injects runtime paths. Key config fields:

- `trainset_3d`, `trainset_2d`, `trainset_humandata`: lists of dataset class names (dynamically imported via `exec()` in `common/base.py`)
- `data_strategy`: `'balance'` (upsample to `total_data_len`) or `'concat'`
- `encoder_config_file`: path to ViTPose encoder config under `main/transformer_utils/configs/smpler_x/encoder/`
- `feat_dim`: ViT feature dimension (768=B, 1024=L, 1280=H)
- `fine_tune`: `'backbone'`, `'neck_and_head'`, or `'head'` to freeze parts of the model

## Data

- Most datasets are stored in `dataset/` and preprocessed into [HumanData](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/human_data.md) `.npz` format under `dataset/preprocessed_datasets/`.
- Exceptions (not in HumanData format): AGORA, MSCOCO, MPII, Human3.6M, UBody — these follow the OSX data preparation pipeline.
- Body model files (SMPL/SMPL-X `.pkl`/`.npz`) must be placed at `common/utils/human_model_files/`.
- The `Cache` class in `data/humandata.py` provides fast dataset loading; run `tool/cache/fix_cache.py` to fix absolute paths in cached `.npz` files if data is moved.

## Required External Assets (not in repo)

- ViTPose pretrained weights: `pretrained_models/vitpose_{small,base,large,huge}.pth`
- SMPLer-X pretrained weights: `pretrained_models/smpler_x_{s,b,l,h}32.pth.tar`
- mmdet Faster-RCNN: `pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth` + config
- SMPL-X model files: `common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl` etc.
- SMPL model files: `common/utils/human_model_files/smpl/SMPL_NEUTRAL.pkl` etc.
