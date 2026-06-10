# Diffusion Policy 推理预算分配分析中文报告

## 1. 项目概述

本项目基于机器人模仿学习方法 Diffusion Policy，研究其在推理部署阶段的一个实际问题：在固定计算预算下，机器人应该把更多计算花在动作 denoising 质量上，还是花在更频繁的 replanning 上？

Diffusion Policy 会通过多步 denoising 生成一段 action chunk，然后在 receding-horizon 控制框架中执行其中一部分动作。这里有两个关键推理参数：

- `k`：每次 policy call 的 denoising steps 数量。
- `h`：每次生成 action chunk 后执行多少个控制步再重新规划。

直观上，`k` 越大，每次动作生成质量越高，但计算更贵；`h` 越小，机器人 replanning 越频繁，反应更快，但 policy calls 更多。因此，本项目将推理预算近似定义为：

```text
B ~= k / h
```

项目核心问题是：在相同或相近的 `B` 下，不同的 `(k,h)` 分配是否会导致不同的控制效果？如果会，能否设计一个根据任务阶段自适应选择 `(k,h)` 的 scheduler？

## 2. 复现基础

本项目首先复现了官方 Diffusion Policy Push-T low-dimensional checkpoint。

官方资源包括：

- 官方源码：`official_reproduction/diffusion_policy_source`
- 官方 checkpoint：`official_reproduction/data/epoch=0550-test_mean_score=0.969.ckpt`
- 官方评测 wrapper：`scripts/run_official_pusht_eval.py`

在当前容器中，官方代码需要兼容 Python 3.12、PyTorch 2.7 和 Gym 0.26，因此做了若干兼容补丁，包括：

- 禁用 custom PushT observation space 的 shared-memory batching。
- 适配新版 Gym 的 `reset()` API。
- 修改 Gym vector concatenate 参数顺序。

官方 checkpoint 的 50-seed Push-T 复现结果为：

```text
mean score = 0.9186505414953691
```

checkpoint 文件名中标注的分数为 `0.969`。当前结果低于文件名分数，主要被记录为环境版本差异造成的 reproduction gap，而不是 checkpoint 下载或执行失败。

## 3. 方法设计

本项目的改进不是重新训练 Diffusion Policy，而是在推理阶段研究 `(k,h)` 如何分配。

我们评估了固定 `(k,h)` 网格，例如：

- `(2,1)`
- `(4,2)`
- `(8,4)`
- `(16,8)`

这些点中有一些具有相同或相近的 `B = k/h`，但它们的控制行为可能不同。

在固定网格之外，项目实现了两个 training-free scheduler：

### 3.1 Score-oriented joint scheduler

这个 scheduler 的目标是提高平均 score，同时减少不必要的 policy calls 和动作抖动。

基本规则是：

- free-space approach 阶段：使用稍长 action chunk，例如 `(4,2)`。
- contact/pushing 阶段：使用更频繁 replanning，例如 `(2,1)`。
- near-goal alignment 阶段：使用相对平滑的 chunk。

### 3.2 Safe joint scheduler

safe scheduler 更保守，目标是提高成功率。

它会更早切换到 `(2,1)`，也就是更频繁 replanning，尤其在：

- pusher 和 object 距离较近时；
- 已经发生 contact 时；
- recent progress 变差时；
- 任务进入高风险接触阶段时。

因此，safe scheduler 不一定追求最高平均 score，而是追求更高 reliability。

## 4. Surrogate 实验结果

主 scheduler improvement 实验是在一个 Push-T-style surrogate 环境中完成的。原因是 surrogate runner 更容易系统控制 `(k,h)`，也更适合分析推理预算分配问题。

优化后实验从 8 seeds 扩展到 20 matched seeds。主要结果如下：

| 方法 | Score | Success | Policy Calls | Smoothness Cost |
|---|---:|---:|---:|---:|
| Fixed `(2,1)` baseline | 0.871 | 0.900 | 120.0 | 0.420 |
| `joint_scheduler` | 0.893 | 0.850 | 89.2 | 0.257 |
| `joint_scheduler_safe` | 0.863 | 0.950 | 105.3 | 0.340 |

这些结果说明：

1. `joint_scheduler` 相比固定 `(2,1)` baseline，提高了平均 score，并显著减少 policy calls 和 smoothness cost。
2. `joint_scheduler_safe` 相比固定 `(2,1)` baseline，提高了 success rate，同时仍然减少 policy calls 和 smoothness cost。
3. 两个 scheduler 不是简单地全面碾压 baseline，而是提供了不同的 Pareto operating points。

也就是说：

- 如果更关注平均 score 和效率，可以选择 `joint_scheduler`。
- 如果更关注成功率和稳定性，可以选择 `joint_scheduler_safe`。

## 5. 官方 Checkpoint Fixed Frontier 结果

为了避免 surrogate 结果被过度解释，项目额外实现了官方 checkpoint 的 fixed `(k,h)` frontier runner：

```text
scripts/run_official_kh_grid.py
```

它直接加载官方 Diffusion Policy checkpoint，并测试不同固定 `(k,h)` 组合。

官方 checkpoint 的 20-seed high-budget frontier 结果如下：

| Official `(k,h)` | Score | Success | Policy Calls |
|---|---:|---:|---:|
| `(12,1)` | 0.107 | 0.000 | 300.0 |
| `(25,2)` | 0.155 | 0.000 | 150.0 |
| `(50,4)` | 0.653 | 0.350 | 75.0 |
| `(100,8)` | 0.949 | 0.950 | 38.0 |

这个结果非常重要。它说明官方 pretrained checkpoint 非常依赖足够多的 denoising steps。低 `k` 的时候，即使 replanning 更频繁，表现也无法恢复。

因此，不能简单说：

```text
低 k + 高频 replanning 可以替代高 k denoising。
```

更准确的结论是：

```text
denoising 和 replanning 的最优分配是 model-dependent 的。
```

也就是说，surrogate 中有效的低 `k` scheduler 不能直接无条件迁移到官方 checkpoint。

## 6. 本项目的主要贡献

本项目最大的贡献不是简单地减少 policy calls，也不是单独让动作更平滑。

最大贡献是：

```text
系统分析了 Diffusion Policy 推理阶段 denoising steps 和 replanning horizon 之间的 compute-control tradeoff。
```

具体包括三点：

### 6.1 问题定义

项目将 Diffusion Policy 推理部署中的两个关键变量 `k` 和 `h` 放到同一个预算框架中分析：

```text
B ~= k / h
```

这使得问题不再是单独调 denoising steps，或者单独调 replanning frequency，而是一个联合预算分配问题。

### 6.2 系统实验分析

项目通过 fixed `(k,h)` grid 和 iso-compute frontier 说明：

- 相同 `B` 下，不同 `(k,h)` 表现可能明显不同。
- denoising depth 和 replanning frequency 对 score、success、calls、smoothness 有不同影响。
- contact-rich task 中，频繁 replanning 和动作质量之间存在明显 tradeoff。

### 6.3 Scheduler 和官方 frontier 校准

项目实现了 phase-adaptive scheduler，并在 surrogate 中展示了更好的 score/cost 或 success/cost tradeoff。

同时，官方 checkpoint frontier 说明：

- 官方模型低 denoising 时会明显失败。
- scheduler claim 不能过度推广。
- 最优预算分配依赖模型本身和预算区间。

这使得项目结论更加严谨。

## 7. 为什么不能说已经在官方 checkpoint 上证明 scheduler 优于 baseline

目前项目还不能严格 claim：

```text
我们的 scheduler 在官方 Diffusion Policy checkpoint rollout 上优于官方 baseline。
```

原因是：

1. 官方 checkpoint 上目前跑的是 fixed `(k,h)` frontier，而不是 dynamic scheduler。
2. 官方 eval runner 默认使用固定 inference settings，不是为 episode 内动态改变 `(k,h)` 设计的。
3. surrogate scheduler 常用较低 `k`，但官方 checkpoint 在低 `k` 下表现很差。
4. 要严格证明，需要实现 official dynamic scheduler runner，在每个 episode 内动态选择 `(k,h)`，并和官方 fixed baseline 使用同一批 seeds 对比。

因此，现在可以严谨地说：

```text
我们在 surrogate 中证明了 scheduler 可以改善 compute-control tradeoff；
我们在官方 checkpoint 上证明了 fixed `(k,h)` frontier 中 denoising depth 非常关键；
两者结合说明这个预算分配问题真实存在且值得研究，但官方 checkpoint 上的 dynamic scheduler improvement 还需要进一步实验验证。
```

## 8. 减少 Calls 和动作更平滑的重要性

减少 policy calls 和动作更平滑不是最大贡献本身，但它们是重要的实验指标。

减少 calls 的意义在于：

- 降低推理成本；
- 降低控制延迟；
- 更适合资源受限的机器人硬件；
- 让 Diffusion Policy 更接近实际部署需求。

动作更平滑的意义在于：

- 减少机器人执行器抖动；
- 提升接触过程稳定性；
- 降低真实机器人磨损；
- 可能提高 sim-to-real 可靠性。

但是，如果只减少 calls 或只让动作更平滑，而 score 和 success 下降很多，那并没有价值。

本项目有意义的地方在于：

- `joint_scheduler` 在提高 score 的同时减少 calls 和 smoothness cost。
- `joint_scheduler_safe` 在提高 success 的同时减少 calls 和 smoothness cost。

因此，这些指标是对 compute-control budgeting 观点的支撑证据。

## 9. 整体评价

优化后的项目已经达到优秀 Final Project 的标准。

主要原因是：

- 官方 Diffusion Policy checkpoint 已经跑通。
- 有完整的 fixed `(k,h)` grid 和 scheduler 实验。
- surrogate 实验扩展到 20 matched seeds。
- 新增 safe scheduler，展示 success/cost tradeoff。
- 新增官方 checkpoint fixed frontier，避免过度 claim。
- report、slides、README、figures、audit log 都已更新。
- 项目代码、结果、图表和交付物都有清晰版本记录。

不过，项目仍然有明确限制：

- scheduler improvement 主要来自 surrogate，而不是官方 checkpoint dynamic rollout。
- 官方 checkpoint 运行环境不是原始 Python 3.9 / PyTorch 1.12 环境。
- 没有真实机器人实验。
- 当前 scheduler 是 rule-based，而不是学习得到的。

因此，最准确的最终评价是：

```text
这是一个具有完整复现、系统推理预算分析和诚实 limitation 的优秀 Final Project。
```

它的贡献不是声称打败官方 Diffusion Policy，而是清楚展示了 Diffusion Policy 部署时 denoising 和 replanning 的联合预算分配问题，并用 surrogate scheduler 和官方 frontier 实验证明这个问题值得认真研究。

## 10. 推荐答辩表述

如果老师问“你的最大贡献是什么”，可以回答：

```text
我们的最大贡献是把 Diffusion Policy 的推理阶段看成一个 joint compute-control budgeting 问题。我们系统分析了 denoising steps k 和 replanning horizon h 的 tradeoff，发现相同平均预算下，不同分配会导致明显不同的 score、success、policy calls 和 smoothness。我们进一步实现了 phase-adaptive scheduler，在 surrogate 中改善了 score/cost 或 success/cost tradeoff。同时，我们用官方 checkpoint fixed frontier 校准了结论，发现官方模型强烈依赖高 denoising depth，因此不能简单把低-k scheduler 结论过度推广。
```

如果老师问“你的 scheduler 是否打败了官方 baseline”，可以回答：

```text
还不能这样 claim。我们已经复现了官方 checkpoint，并在官方 checkpoint 上测试了 fixed `(k,h)` frontier；但 dynamic scheduler 的 improvement 目前是在 surrogate 中验证的。官方 frontier 反而说明官方 checkpoint 在低 denoising steps 下会明显失败。因此我们的严谨结论是：scheduler 在 surrogate 中展示了有价值的 compute-control tradeoff，而官方 checkpoint 实验说明这个 tradeoff 是 model-dependent 的。未来工作需要实现 official dynamic scheduler runner 来直接验证官方 checkpoint 上的 scheduler improvement。
```

