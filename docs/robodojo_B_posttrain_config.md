# RoboDojo B 组 Post-train 配置

B 组固定每个 Task 使用 100 条 episode，改变训练任务数量，用于测量任务多样性对后训练效果的影响。Baseline 已完成，本轮只训练 T1–T4。

| 实验 | 训练任务数 | 训练任务 | Episode / Task | 总 Episodes | GPU | Global Batch | Total Steps | 初始化 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| T1 | 2 | Cover Blocks、Play Tic Tac Toe | 100 | 200 | 8 × A800 | 96 | 30,000 | OpenWAM Alpha foundation |
| T2 | 4 | T1 + Match And Pick From Conveyor、Classify Objects | 100 | 400 | 8 × A800 | 96 | 30,000 | OpenWAM Alpha foundation |
| T3 | 6 | T2 + Fold Clothes、Pour Balls Into Vase | 100 | 600 | 8 × A800 | 96 | 30,000 | OpenWAM Alpha foundation |
| T4 | 8 | T3 + Stack Bowls、Organize Table | 100 | 800 | 8 × A800 | 96 | 30,000 | OpenWAM Alpha foundation |

训练实现使用 raw 14D absolute joint action，33 个时刻窗口、32 步 action horizon、视频 stride 4、三视角输入、bf16、DeepSpeed ZeRO-2、AdamW、学习率 1e-4、cosine schedule、5% warmup、video/action loss 权重均为 1。8 卡时每卡 batch 为 4，梯度累积 3，因此实际 global batch 为 8 × 4 × 3 = 96。每 1,000 个 optimizer step 保存一个 checkpoint，并保留 31 个 checkpoint；每个实验使用 seed 42。

数据目录：

- T1：/media/damoxing/datasets/model-scaling/datasets/robodojo10_B/T1
- T2：/media/damoxing/datasets/model-scaling/datasets/robodojo10_B/T2
- T3：/media/damoxing/datasets/model-scaling/datasets/robodojo10_B/T3
- T4：/media/damoxing/datasets/model-scaling/datasets/robodojo10_B/T4

四个目录中的 episode 都复用 Baseline 的固定 100 条/任务子集，并以任务级目录选择实现嵌套关系；没有重新抽样，也没有混入评测任务之外的数据。

启动脚本：

```bash
cd /media/damoxing/datasets/model-scaling/repos/OpenWAM
bash scripts/launch_robodojo_B_8gpu.sh T1
bash scripts/launch_robodojo_B_8gpu.sh T2
bash scripts/launch_robodojo_B_8gpu.sh T3
bash scripts/launch_robodojo_B_8gpu.sh T4
```

AIHC 约束：

- 资源池：benchmark_bj_a800 (cce-pmm1yohj)
- 队列：train22
- 任务名：WAM-B-T1-OpenWAM-SFT、WAM-B-T2-OpenWAM-SFT、WAM-B-T3-OpenWAM-SFT、WAM-B-T4-OpenWAM-SFT
- 每个任务 1 个 replica，8 张 A800，RDMA 开启
- 只使用 train21/train22 分布式实验队列；本轮选择 train22 以容纳四个任务并避免 train21 排队
