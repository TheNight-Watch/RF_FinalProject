# 面向低维 Push-T Diffusion Policy 的 Mask-Aware Temporal Keypoint Imputation

## 摘要

本项目复现了 Diffusion Policy 官方 low-dimensional Push-T checkpoint，并最终选择一个可直接在官方 checkpoint 上验证的 inference-time improvement：**mask-aware temporal keypoint imputation**。官方 Push-T keypoint runner 会提供 keypoint 坐标和 `obs_mask`，但 low-dimensional policy 的 `predict_action` 只读取 `obs_dict["obs"]`，不会使用 `obs_mask`。在真实部署中，如果 keypoint detector 因遮挡无法返回某些点，常见的 naive 处理会把缺失 keypoint 填成 0，这会给在 full keypoint visibility 上训练的 checkpoint 制造 OOD 输入。

我们实现了一个不改模型、不重训、不改 sampler 的 test-time observation wrapper，在 policy normalizer 之前利用 mask 对缺失 keypoints 做填补。20 个 matched official Push-T seeds 的结果显示，在 50% keypoint visibility 下，zero-fill 只有 `0.177` score 和 `0.00` success，而 carry-forward 达到 `0.901 / 0.75`，linear extrapolation 达到 `0.875 / 0.85`。在 25% visibility 下，linear 从 zero-fill 的 `0.134 / 0.00` 提升到 `0.668 / 0.35`。

本报告的 claim 很窄：这是针对 low-dimensional Push-T keypoint dropout 的免训练部署 wrapper，不是通用视觉 Diffusion Policy 遮挡算法。

## 1. 项目背景与重新定位

课程要求是复现一篇 robotics paper，并在其基础上实现 improvement idea。本项目复现对象是 Diffusion Policy 的官方 Push-T low-dimensional checkpoint。

项目最初探索过 dynamic scheduler 和 DDIM sampler replacement。经过 review 后，这两个方向都不适合作为最终主线：

- dynamic scheduler 的正结果主要来自手写 surrogate，不能证明官方 checkpoint 上有效；
- DDIM 是已有经典 sampler，Diffusion Policy 相关设置中也已经出现，不能包装成原创 improvement。

最终主线改为一个更窄但更诚实的 deployment robustness 问题：

```text
官方 lowdim Push-T runner 提供 keypoint visibility mask，
但默认 lowdim policy 部署路径丢弃该 mask。
当真实 detector-style dropout 导致 keypoint 缺失并被 zero-filled 时，
官方 checkpoint 会严重退化。
一个免训练的 mask-aware temporal imputation wrapper 可以显著恢复性能。
```

## 2. 源码审计：为什么这个 failure mode 成立

这个项目最重要的安全性来自源码核实，而不是只看结果。

### 2.1 Policy 不读取 `obs_mask`

官方 lowdim policy 的 `predict_action` 核心逻辑是：

```python
nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
```

它只使用 `obs_dict["obs"]`，没有读取 `obs_mask`。

### 2.2 Runner 实际上传入了 `obs_mask`

官方 Push-T keypoint runner 会构造：

```python
"obs": obs[..., :self.n_obs_steps, :Do]
"obs_mask": obs[..., :self.n_obs_steps, Do:] > 0.5
```

也就是说，mask 信息在 runner 层是存在的，但在进入 lowdim policy 的过程中没有被使用。

更精确的说法不是“policy 本来应该吃 mask”，而是：

```text
默认 lowdim deployment path 在 policy inference 前丢弃了可见性信息，
且没有做缺失 keypoint 填补。
```

### 2.3 训练时输入是什么

训练 dataset 读取 replay buffer 中的：

- `keypoint`
- `state`
- `action`

然后把 `keypoint.reshape(...)` 和 agent position 拼成 20 维 observation。dataset 没有 mask channel，也没有 keypoint dropout augmentation。

训练配置中：

```yaml
keypoint_visible_rate: 1.0
```

因此官方 checkpoint 是在 full visibility keypoint observation 上训练的。我们在 evaluation 中把 masked keypoint 置零，是为了模拟真实 detector 遮挡时“不返回坐标/返回零”的部署输入。这个输入对 checkpoint 来说是 OOD，所以 zero-fill 崩溃是合理的；我们的 wrapper 解决的是这个部署缺口。

## 3. 方法：Mask-Aware Temporal Observation Wrapper

我们实现了 `DeploymentWrapper`，只在 policy inference 前处理 observation：

- 不改 checkpoint weights；
- 不重新训练；
- 不改 sampler；
- 不改 action horizon；
- 不改 Push-T dynamics；
- 在 normalizer 之前、原始 0-512 keypoint 坐标空间做填补。

对于每个 keypoint 坐标，如果 mask 表示可见，就使用真实坐标；如果不可见，比较以下策略：

| 方法 | 含义 |
|---|---|
| zero-fill | 直接把不可见 keypoint 置 0 |
| mean prior | 用 checkpoint normalizer 中的训练集均值填补 |
| frame hold | 使用最近一次 18 个 keypoint 坐标全部可见的整帧 |
| carry-forward | 每个 keypoint 坐标独立使用最近一次可见值 |
| linear | 每个 keypoint 坐标独立用最近两次可见值做线性外推，并 clamp 到 `[0,512]` |
| oracle | 保留真实坐标，作为不可部署上界 |

首步如果某个 keypoint 从未可见，fallback 是 training mean prior，不是 0。carry-forward 和 linear 都是逐 keypoint/逐坐标更新，不是整帧一起 hold。

## 4. 实验设置

在测试 improvement 之前，我们先用官方 runner 对公开 checkpoint 做了复现 sanity check：

| Evaluation | Mean score | Notes |
|---|---:|---|
| Local 50-seed DDPM reproduction | 0.919 | compatibility environment |
| Checkpoint filename reference | 0.969 | released checkpoint filename |

这里的差距记录为环境版本 gap：本项目在 Python 3.12 / PyTorch 2.7 / Gym 0.26 兼容环境中运行官方代码，而不是原始训练/评测栈。这个数字应该放进正文，因为 reproduction 是课程要求的一半。

正式实验设置：

- checkpoint：官方 low-dimensional Push-T Diffusion Policy checkpoint；
- sampler：DDIM，`(k,h)=(100,8)`；
- seeds：20 matched official test seeds；
- visibility：50% 和 25%；
- mask：官方环境产生的 iid per-keypoint visibility mask；
- metric：Push-T mean score 和 success rate，其中 success 定义为 score `>= 0.95`；
- 统计：paired bootstrap confidence interval。

这里使用 DDIM 只是为了得到 deterministic、稳定的评估设置，不是最终 contribution。最终 contribution 是 mask-aware observation wrapper。这个 failure mode 在 sampler 之前发生：deployment path 在 policy inference 前丢弃 `obs_mask`，所以源码层面的问题与 DDPM/DDIM sampler 无关；本文的正式数值表是在 DDIM 下报告的。

所有方法使用同一批 seeds。由于 env 用相同 seed 初始化，mask 随机序列是可复现的。不同 imputation 会改变动作轨迹，因此后续状态不同，但这正是 policy deployment 比较应包含的闭环差异。

## 5. 主要结果

| visible rate | method | score | success >=0.95 | delta vs zero | 95% CI |
|---:|---|---:|---:|---:|---:|
| 50% | zero-fill | 0.177 | 0.00 | 0.000 | - |
| 50% | mean prior | 0.405 | 0.25 | +0.228 | [+0.058, +0.405] |
| 50% | frame hold | 0.403 | 0.20 | +0.226 | [+0.060, +0.404] |
| 50% | carry-forward | 0.901 | 0.75 | +0.725 | [+0.586, +0.846] |
| 50% | linear | 0.875 | 0.85 | +0.698 | [+0.544, +0.835] |
| 50% | oracle | 0.900 | 0.90 | +0.723 | [+0.571, +0.854] |
| 25% | zero-fill | 0.134 | 0.00 | 0.000 | - |
| 25% | mean prior | 0.257 | 0.00 | +0.123 | [+0.041, +0.214] |
| 25% | frame hold | 0.257 | 0.00 | +0.123 | [+0.040, +0.216] |
| 25% | carry-forward | 0.575 | 0.20 | +0.440 | [+0.290, +0.587] |
| 25% | linear | 0.668 | 0.35 | +0.533 | [+0.378, +0.685] |
| 25% | oracle | 0.900 | 0.90 | +0.766 | [+0.627, +0.881] |

结果说明：

1. zero-fill 会让官方 checkpoint 几乎完全失败。
2. mean prior 和 frame hold 比 zero-fill 好，但提升有限。
3. 逐 keypoint 的 temporal imputation 是关键，carry-forward 和 linear 都显著强于 zero-fill。
4. 50% visibility 下 temporal imputation 接近 oracle 上界。
5. 25% visibility 下仍有明显恢复，但距离 oracle 还有差距，说明重度遮挡没有完全解决。

## 6. Linear vs Carry-Forward

25% visibility 下，linear 的均分和 success 高于 carry-forward：

- linear：`0.668 / 0.35`
- carry-forward：`0.575 / 0.20`

但是 paired bootstrap CI 跨 0：

- score delta `linear - carry_forward = +0.093`，95% CI `[-0.125, +0.305]`
- success delta `+0.15`，95% CI `[-0.10, +0.40]`

因此不能写“linear 显著优于 carry-forward”。更安全的表述是：

```text
Linear extrapolation shows a favorable trend under heavier dropout, but this 20-seed evaluation does not prove strict dominance over carry-forward.
```

## 7. 符合 Final Project 要求的原因

课程要求：

```text
Reproduce an existing robotics paper and implement some improvement ideas on top of it.
```

本项目满足：

### Reproduction

- 复现 Diffusion Policy 官方 lowdim Push-T checkpoint；
- 跑通官方 runner；
- 记录了环境版本导致的 reproduction gap。

### Improvement

- 实现一个免训练 inference-time observation wrapper；
- 解决官方 lowdim deployment path 没有利用 mask 的问题；
- 在 official checkpoint 上直接验证；
- 不依赖 surrogate。

### Evidence

- 20 matched seeds；
- 多个合理 baseline；
- paired bootstrap CI；
- 明确 oracle 上界；
- 明确 limitation。

## 8. 不能写的 claim

以下说法不能写：

- “我们发明了 temporal imputation。”
- “我们解决了通用视觉 Diffusion Policy 遮挡鲁棒性。”
- “我们在 clean full-observation setting 下提升了官方 checkpoint。”
- “linear 显著优于 carry-forward。”
- “这个方法适用于所有机器人任务。”

推荐最终 claim：

```text
We identify a mask-handling failure mode in the official low-dimensional Push-T Diffusion Policy deployment path and show that a training-free, mask-aware temporal observation wrapper substantially improves robustness under detector-style keypoint dropout.
```

## 9. Limitations

当前最快可提交版本仍有以下 limitation：

1. 目前遮挡模型是官方环境的 iid per-keypoint dropout。真实 detector 遮挡更可能是连续成片的 temporally correlated dropout。
2. 方法只验证在 low-dimensional Push-T keypoint observation 上，不是 RGB visual policy。
3. 25% visibility 下 temporal imputation 仍明显低于 oracle，说明重度遮挡没有完全解决。
4. 当前官方复现环境是兼容环境，不完全等同于原始 Python/PyTorch/Gym 版本。

这些 limitation 不推翻当前 claim，但必须在报告和答辩中主动说明。

## 10. 最终答辩表述

如果老师问“你的 contribution 是什么”，可以回答：

```text
我们发现官方 lowdim Push-T Diffusion Policy 的部署路径中存在一个 mask-handling failure mode：runner 提供 keypoint visibility mask，但 policy inference 实际只使用 keypoint 坐标。如果真实 detector 遮挡导致 keypoint 被 zero-filled，官方 checkpoint 会严重退化。我们实现了一个不需要训练的 mask-aware temporal observation wrapper，在 normalizer 前对缺失 keypoint 做逐点时序填补，并在 20 个 matched official seeds 上证明它显著优于 zero-fill、mean prior 和 frame hold。
```

如果老师问“这是不是通用算法”，要回答：

```text
不是。我们的 claim 很窄，只针对 low-dimensional Push-T keypoint dropout deployment。它不是通用视觉 Diffusion Policy 遮挡算法，也不 claim clean setting 更好。
```

如果老师问“为什么没补连续遮挡”，要回答：

```text
当前版本使用官方环境内置的 iid keypoint visibility mask。连续遮挡更接近真实 detector failure，是下一步应该补的实验；因此我们把它写成 limitation，没有把结果过度外推到真实视觉遮挡。
```

## 11. 结论

最快可提交版本的最终主线是：

```text
Official Diffusion Policy Push-T reproduction
+ training-free mask-aware temporal keypoint imputation
+ official-checkpoint robustness evaluation under keypoint dropout.
```

这是目前项目中最稳的版本：贡献窄，但证据直接、baseline 完整、claim 边界清楚，符合 final project 的 reproduction + improvement 要求。
