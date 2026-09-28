# Local OpenWAM CUDA environment

Validated on 2026-09-26 for the local A800 container:

- Environment: `/mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124`
- Python: 3.12.8
- PyTorch: 2.5.1+cu124
- Transformers: 5.17.0
- Diffusers: 0.40.0
- DeepSpeed: 0.18.9
- OpenWAM: editable checkout at `repos/OpenWAM`
- Private cuDNN: 9.5.1 at `openwam_cu124/cudnn/9.5.1/lib`

The environment reuses the local `aihc-miniforge` CUDA/PyTorch packages and
contains a private copy of the OpenWAM-specific Python packages. It does not
modify `caliball_cu121` or the previous `openwam` venv. The launchers prepend
the private cuDNN directory because the inherited cuDNN 9.1 installation has
an inconsistent core/graph library pair.

Run the ego+robot one-step smoke on GPU 1:

```bash
cd /mnt/world_foundational/datasets/model-scaling/repos/OpenWAM
bash scripts/smoke_figure10_pku_local.sh
```

Select another GPU with `OPENWAM_SMOKE_GPU=0`. The smoke uses batch size 1,
zero dataloader workers, CPU optimizer offload, gradient checkpointing, one
training step, disabled W&B, and no checkpoint writes.

Pass `robot-only`, `ego2robot-ego`, `ego2robot-robot`, or `ego-robot` as the
first argument to select a Figure 10 recipe. The ego2robot robot stage requires
`OPENWAM_EGO_STAGE_CKPT` to point at a self-contained completed stage directory.

All Figure 10 pretraining paths completed a real one-step smoke on GPU 1 on
2026-09-26. Each run covered data decoding, forward, backward, DeepSpeed
gradient clipping, and the optimizer step with PyTorch 2.5.1+cu124 and the
private cuDNN 9.5.1 runtime:

| Recipe | Result | One-step metrics |
| --- | --- | --- |
| Robot only, 600h | pass | loss 0.7686, video 0.6386, action about 0.13 |
| Ego2robot, ego 350h | pass | loss/video 0.6009, action 0.0 |
| Ego2robot, robot 250h | pass | loss 0.4655, video 0.4441, action 0.0214 |
| Ego + robot | pass | loss/video 0.5619 for the sampled ego item |

The ego2robot robot-stage smoke also loaded the self-contained
`study/two-stage-clean/checkpoint_step_30000.safetensors` before training.
