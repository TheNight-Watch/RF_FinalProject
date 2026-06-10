# Diffusion Policy Push-T 官方 Checkpoint 采样器改进中文报告

## 1. 重新定位后的项目主线

经过 Claude review 和相关工作审计后，原来的 dynamic scheduler 主张已经降级。现在项目的主线改为一个更稳、能直接在官方 checkpoint 上验证的 improvement：

```text
在官方 Diffusion Policy Push-T checkpoint 上，将原始 DDPM sampler 替换为 DDIM deterministic sampler，并比较其在不同 `(k,h)` 设置下的表现。
```

这里：

- `k`：每次 policy call 的 denoising steps。
- `h`：每次 action chunk 执行多少步后 replanning。
- DDPM：官方 checkpoint 默认使用的 ancestral diffusion sampler。
- DDIM：经典 deterministic diffusion sampler，不需要重新训练模型。

这个主张比原来的 surrogate scheduler 更稳，因为它：

1. 直接作用在官方 checkpoint 上；
2. 不依赖手写 surrogate 环境；
3. 不需要训练新模型；
4. 有 matched seeds 的官方 Push-T rollout 结果；
5. 不声称算法原创，只 claim plug-and-play sampler replacement 在部分设置下有效。

## 2. 官方 Diffusion Policy 复现

本项目已经跑通官方 Diffusion Policy low-dimensional Push-T checkpoint。

关键文件：

- 官方源码：`official_reproduction/diffusion_policy_source`
- 官方 checkpoint：`official_reproduction/data/epoch=0550-test_mean_score=0.969.ckpt`
- 官方评测 wrapper：`scripts/run_official_pusht_eval.py`

官方 checkpoint 的 50-seed DDPM 复现结果为：

```text
mean score = 0.9186505414953691
```

checkpoint 文件名中标注的分数为 `0.969`。当前分数低于文件名分数，主要记录为环境版本差异造成的 reproduction gap。

## 3. 新的 Improvement Idea：DDIM Sampler Replacement

原始官方 checkpoint 使用 DDPM scheduler。我们的改进是在推理阶段替换 sampler：

```python
policy.noise_scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
```

其他内容保持不变：

- 模型权重不变；
- normalizer 不变；
- observation history 不变；
- Push-T 环境不变；
- test seeds 匹配；
- `(k,h)` 设置匹配。

也就是说，这是一个真正的 inference-time plug-and-play improvement。

实现位置：

```text
scripts/run_official_kh_grid.py --sampler ddim
```

分析脚本：

```text
scripts/analyze_official_sampler_comparison.py
```

## 4. 官方 Checkpoint 上的正式结果

我们比较了 DDPM 和 DDIM 在相同 `(k,h)` 下的 20-seed official Push-T 表现。

| `(k,h)` | DDPM Score | DDIM Score | Delta | DDPM Success | DDIM Success |
|---|---:|---:|---:|---:|---:|
| `(25,2)` | 0.155 | 0.900 | +0.745 | 0.000 | 0.800 |
| `(50,4)` | 0.653 | 0.929 | +0.276 | 0.350 | 0.750 |
| `(100,8)` | 0.949 | 0.900 | -0.048 | 0.950 | 0.900 |

paired bootstrap 95% confidence interval：

- `(25,2)`: `[+0.593, +0.870]`
- `(50,4)`: `[+0.056, +0.488]`
- `(100,8)`: `[-0.152, +0.007]`

这个结果很有价值。

它说明：

1. DDIM 在低/中 denoising 设置下显著优于 DDPM。
2. `(25,2)` 下，DDPM 几乎失败，而 DDIM 达到 `0.900` score 和 `0.800` success。
3. `(50,4)` 下，DDIM 把 score 从 `0.653` 提高到 `0.929`，success 从 `0.350` 提高到 `0.750`。
4. 但 DDIM 不是所有设置都更好，`(100,8)` 下 DDPM 仍然更强。

所以我们的严格 claim 是：

```text
DDIM sampler replacement improves the official checkpoint's low/mid-denoising inference frontier compared with the official DDPM sampler.
```

不能说：

```text
DDIM 全面打败 DDPM。
```

## 5. 为什么这个符合 Final Project 的 Improvement 要求

课程要求是：

```text
Reproduce an existing robotics paper and implement some improvement ideas on top of it.
```

现在我们的项目满足这个要求：

### 5.1 Reproduce

我们复现了 Diffusion Policy Push-T 官方 checkpoint，并跑通官方 runner。

### 5.2 Improvement

我们在官方 checkpoint 上实现了一个 inference-time sampler replacement：

- baseline：官方 DDPM sampler；
- improvement：DDIM deterministic sampler；
- 不重新训练；
- 不换环境；
- 用 matched official Push-T seeds 对比。

### 5.3 Evidence

这个 improvement 在官方 checkpoint rollout 上有直接证据：

- `(25,2)` 大幅提升；
- `(50,4)` 明显提升；
- paired bootstrap CI 为正；
- 结果不是 surrogate。

这比原来的 scheduler 主张强很多。

## 6. 原 Scheduler 工作的处理方式

项目里仍然保留 surrogate scheduler，但它不再是主贡献。

原因是：

- surrogate 手写了 denoising residual；
- surrogate 手写了 stale-plan effect；
- scheduler 在 surrogate 上提升不能证明官方 checkpoint 上也提升；
- VADF 等相关工作已经做了非常接近的 dynamic scheduling。

因此，最终答辩时应该说：

```text
我们原先探索过 scheduler，但经过 review 后不再把它作为主要 claim。最终可靠的 improvement 是官方 checkpoint 上的 DDIM sampler replacement。
```

## 7. 当前项目的最大贡献

现在最大贡献是：

```text
在复现官方 Diffusion Policy Push-T checkpoint 的基础上，验证了 DDIM sampler replacement 可以显著改善低/中 denoising 设置下的官方 checkpoint rollout 表现。
```

更具体地说：

- 官方 DDPM 在 `(25,2)` 和 `(50,4)` 下表现较差；
- DDIM 在相同 `(k,h)`、相同 seeds 下显著改善；
- 这说明 sampler choice 对 Diffusion Policy deployment 很重要；
- 但高 denoising DDPM `(100,8)` 仍然是强 baseline。

## 8. 最终答辩推荐表述

如果老师问“你的 improvement 是什么”，可以回答：

```text
我们的 improvement 是在官方 Diffusion Policy Push-T checkpoint 上做 plug-and-play sampler replacement：把默认 DDPM sampler 替换成 DDIM sampler。这个改动不需要重新训练模型，只改变推理过程。我们在 matched official Push-T seeds 上验证，DDIM 在 `(25,2)` 和 `(50,4)` 这两个低/中 denoising 设置下显著提升 score 和 success，例如 `(25,2)` 从 0.155 提升到 0.900，`(50,4)` 从 0.653 提升到 0.929。
```

如果老师问“是不是全面超过官方 baseline”，要回答：

```text
不是。我们的 claim 是 bounded 的。DDIM 显著改善低/中 denoising frontier，但在 `(100,8)` 这个高 denoising 设置下，DDPM 仍然更强。因此我们不 claim DDIM 全面打败 DDPM，而是 claim DDIM 扩展了低/中 denoising 设置的可用性。
```

如果老师问“之前 scheduler 怎么办”，可以回答：

```text
我们保留 scheduler 作为 synthetic diagnostic，但不作为主要 evidence。经过 review，我们认为 surrogate scheduler 不能证明 official checkpoint improvement，所以最终主线改为官方 checkpoint 上可直接验证的 DDIM sampler replacement。
```

## 9. 最终评价

当前项目比止损前更符合 Final Project 要求。

它现在具备：

- 官方 paper reproduction；
- 官方 checkpoint rollout；
- 明确 implemented improvement；
- matched seed 官方实验；
- paired bootstrap CI；
- 清楚的 limitation；
- 不依赖 surrogate overclaim。

最终定位：

```text
Official Diffusion Policy reproduction + plug-and-play DDIM sampler improvement.
```

