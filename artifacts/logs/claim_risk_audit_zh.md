# Claim Risk Audit: 新三层 Argument 可行性审计

日期：2026-06-10

## 结论摘要

本次重新审计后，原先“phase-adaptive scheduler 联合调度 `k` 和 `h` 是主要创新”的表述必须止损。

原因是相关工作中已经出现了非常接近的方向：

- **VADF: Vision-Adaptive Diffusion Policy Framework for Efficient Robotic Manipulation** 明确提出 inference 阶段用 VLM 做 stage-wise dynamic scheduling，并同时分配 `n_action_steps` 和 `num_inference_steps`。
- **RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning** 已经针对 Diffusion Policy 的高频 replanning 问题提出 training-free 机制。
- **Adaptive Online Replanning with Diffusion Models** 已经系统研究 diffusion planner 什么时候 replanning，以及 replanning 与 diffusion steps 的计算代价。
- **TIDAL** 和 **RTI-DP** 等近期工作也在研究把推理计算预算重新分布到控制时间轴上，以提高反馈频率或降低延迟。

因此，本项目不能再 claim：

```text
We are the first to jointly schedule denoising steps and action horizon.
```

也不能把 surrogate scheduler 的结果作为 Diffusion Policy 方法创新的 headline。

## 已查到的关键相关工作

### 1. VADF: Vision-Adaptive Diffusion Policy Framework for Efficient Robotic Manipulation

URL: https://arxiv.org/html/2604.15938v1

相关性：非常高。

VADF 在 inference 端提出 HVTS，使用 task decomposition 和 stage classification 来动态分配：

- `n_action_steps`
- `num_inference_steps`

其 appendix B.3 明确要求 VLM 对每个 stage 输出：

```text
n_action_steps
num_inference_steps
```

并且正文声称其 dynamic scheduling 可以根据阶段调节 inference budget。它还在 Push-T 等任务上报告 dynamic scheduling 的效果。

影响：

- 我们的第 3 层“stage-adaptive scheduler 同时调 `k` 和 `h`”不再具有原创性。
- 如果继续做 official dynamic scheduler，只能作为复现/对照/独立验证，不能作为主要创新。

### 2. RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning

URL: https://arxiv.org/abs/2503.04051

相关性：高。

RA-DP 关注 Diffusion Policy 的高频 replanning 能力。它通过 action queue 机制在 denoising 过程中更新可执行动作，实现 training-free high-frequency replanning。

影响：

- “Diffusion Policy replanning frequency 受 iterative sampling 限制”这个问题已经被明确提出。
- 我们不能把“高频 replanning 对 DP 重要”作为独立新发现。

### 3. Adaptive Online Replanning with Diffusion Models

URL: https://openreview.net/forum?id=jhs8F63xI6

相关性：中高。

该工作研究 diffusion planner 中何时需要 replanning，并用 likelihood 判断 existing plan 是否需要更新。它也讨论 replanning 策略与 diffusion steps 的计算代价。

影响：

- “replanning 频率和 diffusion 计算代价存在 tradeoff”不是空白领域。
- 我们的贡献必须更具体地限定在 official Diffusion Policy Push-T checkpoint 的 empirical budget frontier，而不是泛化成 diffusion replanning 新框架。

### 4. TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control

URL: https://arxiv.org/html/2601.14945v1

相关性：中高。

TIDAL 在 iso-compute 约束下把 action generation 分布到时间轴上，提高 action chunk update frequency。它的思想和“同算力下把计算分配给更频繁反馈还是一次性生成”高度相关。

影响：

- “iso-compute 下重新分配推理计算到控制频率”已有相关表述。
- 我们可以借鉴其 framing，但不能说这个 framing 完全没人做过。

### 5. Real-Time Iteration Scheme for Diffusion Policy

URL: https://arxiv.org/html/2508.05396v1

相关性：中。

RTI-DP 使用 previous prediction 作为 initialization，用较少 denoising steps 做迭代 refinement，以降低推理成本并保持性能。

影响：

- “减少 denoising steps 但保持控制效果”的方向已有大量工作。
- 我们的 `k` 维度不能作为单独创新，只能作为 budget frontier 中的一个轴。

### 6. Unpacking the Individual Components of Diffusion Policy

URL: https://arxiv.org/html/2412.00084v1

相关性：中。

该工作系统分析 action sequence execution 和 receding horizon control 等组件。它指出不同任务对 action horizon/receding horizon 的需求不同。

影响：

- “action execution horizon 会影响任务表现”已有系统研究。
- 我们的第 1 层 framing 仍可讲，但必须承认该工作已分析 DP 组件；我们的区别是围绕 official Push-T checkpoint 的 `num_inference_steps` 与 `n_action_steps` fixed frontier。

## 三层 Argument 重新评级

### 第 1 层：问题框架

原说法：

```text
没人做 joint allocation of k and h。
```

审计结论：不成立。

修正后可说：

```text
Recent work has begun to study adaptive inference and replanning in diffusion policies. In this project, we focus on a narrower empirical question: how the public official Push-T checkpoint responds to fixed `num_inference_steps` and `n_action_steps` frontiers under matched rollout seeds.
```

可保留程度：中。

### 第 2 层：同算力、不同分法导致不同结果

原说法：

```text
同样 B=k/h，不同分配会导致不同结果。
```

审计结论：仍然可验证，而且是当前最稳的主线。

理由：

- 这可以直接在 official checkpoint 上验证。
- 不依赖 hand-designed surrogate。
- 不需要声称方法原创。
- 当前已有 B≈12.5 frontier，但只有一个预算档，需要补 B≈4 和 B≈8。

可保留程度：高。

建议主张：

```text
On the official pretrained Push-T checkpoint, equal or similar amortized compute budgets are not interchangeable: allocating compute to denoising depth versus replanning frequency yields sharply different score/success frontiers.
```

### 第 3 层：dynamic scheduler 在官方 checkpoint 上打败固定基线

原说法：

```text
我们的 dynamic scheduler 能优于 fixed baseline。
```

审计结论：当前不成立，且原创性不足。

继续条件：

只有在以下两个条件同时成立时，才值得继续做：

1. official checkpoint 的 state-sensitivity probe 证明 episode 内存在异质性，即某些状态低 `k` 足够，某些状态需要高 `k`；
2. 设计的 scheduler 不是复刻 VADF/RA-DP，而是作为 independent verification 或 class project extension，且 claim 明确降级。

若第 1 条不成立，应立即止损，不做 scheduler superiority claim。

## 修正后的项目主线

建议从现在开始将项目主线改为：

```text
Official Diffusion Policy Push-T checkpoint inference-budget sensitivity analysis.
```

中文：

```text
官方 Diffusion Policy Push-T checkpoint 的推理预算敏感性分析。
```

核心贡献应改成：

1. 复现官方 Push-T checkpoint。
2. 在真 checkpoint 上系统测试多条 fixed `(k,h)` iso-budget frontier。
3. 验证同等预算下 denoising depth 与 replanning frequency 的分配不是等价的。
4. 说明低 denoising 高频 replanning 不能替代足够 denoising。
5. 将 surrogate scheduler 降级为 toy diagnostic 或 appendix，不再作为主证据。

## 下一步实验闸门

### Gate A：补第 2 层官方 frontier

必须跑：

```text
B≈4:  (8,2),  (16,4), (32,8)
B≈8:  (16,2), (32,4), (64,8)
B≈12.5: 当前已有 (12,1), (25,2), (50,4), (100,8)
```

若结果显示不同 `(k,h)` 的分配确实有明显差异，则第 2 层成立。

### Gate B：官方 state-sensitivity probe

如果要进入第 3 层，必须先证明 official checkpoint episode 内存在异质性。

候选 probe：

- 同一 observation 下比较 low-k 与 high-k action discrepancy。
- 同一 observation 下多采样 action disagreement。
- 按 free/contact/near-goal phase 统计 discrepancy。
- 检查 low-k 失败是否集中在某些阶段。

如果 signal 基本平坦，或所有阶段都强依赖高 `k`，则第 3 层止损。

## 当前严禁使用的表述

以下表述不应再出现在最终报告或答辩中：

```text
We propose a novel joint scheduler for denoising and replanning.
```

```text
The scheduler improves Diffusion Policy.
```

```text
Low-k frequent replanning can replace high-k denoising.
```

```text
No prior work studies joint denoising/action-horizon scheduling.
```

## 当前允许使用的表述

可以使用：

```text
This project provides an empirical inference-budget sensitivity analysis of the official Diffusion Policy Push-T checkpoint.
```

```text
The official checkpoint frontier shows that denoising depth is critical, and that equal nominal budgets can produce very different score/success outcomes depending on how compute is allocated.
```

```text
The synthetic scheduler result is retained only as a toy diagnostic and is not used as evidence of official checkpoint improvement.
```

## 新的可成立 Improvement：DDIM Sampler Replacement

在止损 dynamic scheduler 主线后，项目补充了一个更稳的 improvement idea：

```text
在官方 Diffusion Policy Push-T checkpoint 上，将默认 DDPM sampler 替换为 DDIM sampler。
```

这个方向不是算法原创，DDIM 是已有经典 diffusion sampler；但它符合课程要求中的 “implement some improvement ideas on top of it”，因为：

- 直接作用于复现出的官方 checkpoint；
- 不依赖 surrogate；
- 不需要重新训练；
- 使用 matched official Push-T seeds 评测；
- 有 paired bootstrap 置信区间；
- claim 被严格限定在 low/mid-denoising frontier。

正式 20-seed 结果：

| `(k,h)` | DDPM score | DDIM score | delta | DDPM success | DDIM success |
|---|---:|---:|---:|---:|---:|
| `(25,2)` | `0.155` | `0.900` | `+0.745` | `0.000` | `0.800` |
| `(50,4)` | `0.653` | `0.929` | `+0.276` | `0.350` | `0.750` |
| `(100,8)` | `0.949` | `0.900` | `-0.048` | `0.950` | `0.900` |

允许使用的新主张：

```text
DDIM sampler replacement improves the official checkpoint's low/mid-denoising inference frontier compared with the official DDPM sampler.
```

仍然禁止使用的过度主张：

```text
DDIM universally beats DDPM.
```

```text
DDIM beats the best high-denoising official baseline.
```

```text
The earlier dynamic scheduler improves the official checkpoint.
```
