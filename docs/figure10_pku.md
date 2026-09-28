# PKU Figure 10 training recipes

These configs reproduce the OpenWAM Figure 10 pretraining protocol while
replacing its EgoDex/RoboCOIN episode population with the audited PKU subsets
prepared on 2026-09-26. OpenWAM does not publish a standalone Q5 pretraining
YAML: common pretraining hyperparameters below are taken from the released
foundation-model `config.yaml` and paper Table 9, while the 600-hour recipe and
`action_sees_video` mask come from the Figure 10 study protocol. The released
30k-step study configs are downstream RoboTwin SFT configs and are not reused
as pretraining configs here.

| Recipe | Data | Config |
| --- | --- | --- |
| Robot only | 600.060 h robot | `figure10_pku_robot_only` |
| Ego to robot, stage 1 | 350.004 h ego | `figure10_pku_ego2robot_ego` |
| Ego to robot, stage 2 | 250.035 h robot | `figure10_pku_ego2robot_robot` |
| Ego + robot | 350.004 h ego + 250.035 h robot | `figure10_pku_ego_robot` |

The released Figure 10 pretraining checkpoint configs fix the Wan2.2-TI2V-5B
dual-system model, ActionDiT, `action_sees_video` mask, bf16, ZeRO-2, full-layer
gradient checkpointing without CPU activation/optimizer offload, 33-frame
clips, video stride 4, 384x320 multiview input, and
ColorJitter(0.2, 0.2, 0.2, 0.0). The PKU reader emits the same 80-D action head
contract. Robot action targets use absolute source-frame EEF20
(`xyz + rot6d + gripper`) and quantile normalization computed from the exact
selected robot population. Base-EE sources remain in robot base coordinates;
the UR `camera_ee` sources remain in their calibrated camera coordinates, and
their proprio poses are transformed with the same world-to-camera extrinsic.
Rot6d dimensions remain unnormalized; grippers are canonicalized to
`0=closed, 1=open`. Missing source dimensions are represented by explicit
action/proprio masks rather than zero-valued loss targets. Ego samples are
video-only and mask action/proprio supervision.

The robot600 statistics used by robot-only and the robot250 statistics used by
ego+robot and ego2robot stage 2 are stored in their respective subset
directories as `pku_eef_normalization_stats.npy`.
They are bound to the pointer index SHA256 and are rejected if the index,
frame policy, action representation, or gripper convention changes. Regenerate
them with:

```bash
python -m openwam.dataloader.utils.stats_computation.pku_native30_stats_computation \
  --index /mnt/world_foundational/datasets/model-scaling/manifests/pku/scaling_subsets_20260926/robot_250h_20260926/index.json \
  --output /mnt/world_foundational/datasets/model-scaling/manifests/pku/scaling_subsets_20260926/robot_250h_20260926/pku_eef_normalization_stats.npy
```

```bash
cd /mnt/world_foundational/datasets/model-scaling/repos/OpenWAM
export OPENWAM_WAN22_PATH=/mnt/world_foundational/datasets/model-scaling/weights/shared/Wan2.2-TI2V-5B

# Read-only environment/GPU/weight/index check; decodes one real sample.
OPENWAM_PYTHON=/absolute/path/to/python \
OPENWAM_PREFLIGHT_ONLY=1 OPENWAM_PREFLIGHT_SAMPLE=1 \
bash scripts/train_figure10_pku.sh robot-only

bash scripts/train_figure10_pku.sh robot-only
bash scripts/train_figure10_pku.sh ego2robot-ego
export OPENWAM_EGO_STAGE_CKPT=/absolute/path/to/completed/ego2robot_ego_stage
bash scripts/train_figure10_pku.sh ego2robot-robot
bash scripts/train_figure10_pku.sh ego-robot
```

The launcher always runs the read-only preflight before `torchrun`. It never
creates an environment or installs/downloads packages. Set `OPENWAM_PYTHON`
to the audited training interpreter; the launcher then uses that same Python
for preflight and distributed training.

The two-stage recipe intentionally starts stage 2 through
`training.finetune_ckpt_path`; this reloads the completed ego-stage weights and
starts a fresh optimizer/scheduler for the robot stage.

## Four-node, eight-GPU training

Run the experiment-specific PyTorch launchers on all four nodes. They share a
common `torch.distributed.run` implementation which enforces four nodes and
eight visible GPUs per node. The launchers retain bf16, ZeRO-2, full-layer
gradient checkpointing, and one epoch. Batch size is an explicit hardware
adaptation and is reported by the launcher rather than presented as an
unchanged official value.

```bash
cd /mnt/world_foundational/datasets/model-scaling/repos/OpenWAM
export MASTER_ADDR=10.0.0.10  # address of node rank 0
export MASTER_PORT=29500

# Run the same experiment command on every node, changing only NODE_RANK.
NODE_RANK=0 bash scripts/train_figure10_robot_only_4n8g.sh
NODE_RANK=1 bash scripts/train_figure10_robot_only_4n8g.sh
NODE_RANK=2 bash scripts/train_figure10_robot_only_4n8g.sh
NODE_RANK=3 bash scripts/train_figure10_robot_only_4n8g.sh
```

The three experiment entry points are:

```bash
# Robot-only 600h.
NODE_RANK=<0-3> bash scripts/train_figure10_robot_only_4n8g.sh

# Ego2robot stage 1: ego 350h.
NODE_RANK=<0-3> bash scripts/train_figure10_ego2robot_4n8g.sh ego

# Ego2robot stage 2: robot 250h. Set this on all four nodes.
export OPENWAM_EGO_STAGE_CKPT=/shared/path/to/ego2robot_ego_stage
NODE_RANK=<0-3> bash scripts/train_figure10_ego2robot_4n8g.sh robot

# Ego + robot joint training.
NODE_RANK=<0-3> bash scripts/train_figure10_ego_robot_4n8g.sh
```

`NCCL_SOCKET_IFNAME` may be set by the cluster launcher when its network
interface is known; the scripts deliberately do not hardcode one. Hydra
overrides can be appended to any command.

For the 600-hour ego+robot run on a distributed platform, the complete
single-window environment and launch command is available as
`scripts/launch_figure10_ego_robot_600h_4n8g.sh`. Its checkpoint base is
`/media/damoxing/ckp/openwam/figure10_ego_robot_600h_1epoch/<RUN_ID>/`.

On distributed platforms, the common launcher also accepts `HOST_NUM` as the
node count and `RANK` as the node rank, in addition to `NNODES` and
`NODE_RANK`. This allows one identical command to be installed on all nodes.

For the 8-node, 8-GPU-per-node, per-GPU batch-32 ego+robot run, use the same
single command on every platform node:

```bash
bash scripts/launch_figure10_ego_robot_600h_bs32_8n8g.sh
```

The matching robot-only and automatic two-stage ego2robot launchers are:

```bash
# Robot-only: 600h robot, one epoch.
bash scripts/launch_figure10_robot_only_600h_bs32_8n8g.sh

# Ego2robot: automatically run 350h ego and then warm-start 250h robot.
bash scripts/launch_figure10_ego2robot_600h_bs32_8n8g.sh
```

The platform must provide `HOST_NUM=8`, `RANK=0..7`, `MASTER_ADDR`, and
`MASTER_PORT`. The effective global batch is 2048, training runs for one epoch,
progress and W&B metrics are emitted every 20 steps, and model checkpoints are
still saved every 2000 steps.
