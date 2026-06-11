# Innovation Search Audit

Date: 2026-06-11 UTC

## Purpose

在 Claude 指出 DDIM 并非新方法后，本轮重新寻找可作为 final project 增量的方向。原则是先做相关工作审计，再做小规模 official-checkpoint pilot；没有信号或撞已有工作的方向直接止损。

## 已排除方向

### 1. DDIM sampler replacement as an original improvement

结论：不能作为原创 improvement。

原因：

- Diffusion Policy 原论文/官方配置已经包含 DDIM inference setting。
- 官方源码中存在 `train_diffusion_unet_ddim_lowdim_workspace.yaml`，其 `noise_scheduler` 已为 `DDIMScheduler`，`num_inference_steps: 8`。
- 当前 DDIM 结果仍可作为 reproduction-side sampler audit，但不能写成“我们提出 DDIM 替换 DDPM”。

### 2. Chunk overlap / continuity consistency

结论：不能作为原创方向。

原因：

- 检索到 RTC、Legato、DCDP 等非常近的工作，已经围绕 action chunk overlap、continuity、latency-aware execution、dynamic corrections 展开。
- 相关链接：
  - RTC: https://openreview.net/forum?id=UkR2zO5uww
  - Legato / Native Continuation: https://arxiv.org/html/2602.12978
  - DCDP: https://arxiv.org/pdf/2603.01953

### 3. DDIM stochasticity calibration via `eta`

结论：小样本 pilot 没有正向信号，暂不作为主贡献。

实现：

- `scripts/run_official_kh_grid.py` 增加 `--ddim-eta`，只在 `--sampler ddim` 时传入 `policy.kwargs["eta"]`。

4-seed official Push-T smoke results:

| eta | `(25,2)` score | `(25,2)` success | `(50,4)` score | `(50,4)` success |
|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 1.00 | 0.996 | 1.00 |
| 0.25 | 0.983 | 0.75 | 0.963 | 0.75 |
| 0.50 | 0.819 | 0.75 | 0.989 | 1.00 |
| 0.75 | 0.833 | 0.75 | 0.978 | 1.00 |
| 1.00 | 0.564 | 0.50 | 0.993 | 1.00 |

判断：

- `eta=0` 已经最强。
- 较大随机性会显著伤害 `(25,2)`。
- 这更适合写成 negative probe，不适合作为 headline innovation。

### 4. Target-rate limiting action filter

结论：小样本 pilot 没有足够正向信号，暂不作为主贡献。

实现：

- `scripts/run_official_kh_grid.py` 增加 `--postprocess rate_limit --max-target-delta N`。
- 该 wrapper 将 policy 输出的绝对 action target 序列限制为每步目标位移不超过 `N` pixels。

4-seed official Push-T DDPM smoke results:

| max target delta | `(25,2)` score | `(25,2)` success | `(50,4)` score | `(50,4)` success |
|---:|---:|---:|---:|---:|
| 32 | 0.138 | 0.00 | 0.722 | 0.25 |
| 64 | 0.113 | 0.00 | 0.707 | 0.25 |
| 96 | 0.113 | 0.00 | 0.667 | 0.00 |

判断：

- 对 `(25,2)` 基本无救。
- 对 `(50,4)` 可能有一点小样本提升，但不稳定，也没有超过强 DDIM setting。
- 动作过滤本身也有大量泛化相关工作，不能轻易包装为原创算法。

### 5. DPM-Solver / DPM-Solver++ sampler replacement

结论：不能作为原创方向。

原因：

- Falcon 明确包含 “Falcon with DPMSolver”，并给出 diffusion policy with DPMSolver 的采样过程。
- DP3、STEP、Contractive Diffusion Policies 等相关工作也讨论 DPM-Solver / DPM-Solver++ 作为 diffusion policy 加速器或 baseline。
- 因此即使在本项目中接入 DPM-Solver，也只能作为已知 solver baseline，不能作为新的 improvement claim。

相关链接：

- Falcon: https://arxiv.org/html/2503.00339
- STEP: https://arxiv.org/html/2602.08245
- Contractive Diffusion Policies: https://openreview.net/pdf/835fb8d39a2036712e068861f40ead262c41182e.pdf

## 下一步候选

### Task-aware geometric verifier / fallback for Push-T

状态：已完成小样本 pilot，暂不作为主贡献。

想法：

- 利用 Push-T low-dimensional observation 中的 block keypoints 和 agent position，构造一个非常轻量的 geometric sanity check。
- 只在 diffusion action 明显偏离“agent 应移动到 block 后方、沿 goal 方向推动”的几何先验时进行 fallback 或 blending。
- 这不是通用 Diffusion Policy 算法，只能定位为 Push-T task-aware deployment filter。

风险：

- 可能被认为是 task-specific hand-coded controller，不够研究化。
- 需要确认是否已有“Push-T geometric action filter / heuristic correction for Diffusion Policy”相同工作。
- 即使有效，也必须严格写成 task-specific engineering improvement，不能声称通用。

检索结论：

- 没有发现“官方 DP Push-T checkpoint + block-keypoint geometric fallback”的完全相同实验。
- 但 recovery/fallback/guidance/action-filter 类工作很多，例如 REACH、WARPD、policy composition、safe/action filters 等；因此即使有效，也只能定位为 task-specific deployment filter，而不是通用 Diffusion Policy 创新。

实现：

- `scripts/run_official_kh_grid.py` 增加 `--postprocess geom_fallback`。
- 使用当前 low-dimensional observation 中的 9 个 block keypoints 和 agent position：
  - block centroid = 9 个 keypoints 均值；
  - fixed goal center = `(256,256)`；
  - fallback target = block 后方 approach point 或沿 goal direction 的 push target；
  - 当 raw action direction 与 fallback direction 点积为负时，以 `geom_alpha` 混合 fallback；
  - 最后加 target-rate limit。

4-seed official Push-T DDPM smoke results:

| method | `(25,2)` score | `(25,2)` success | `(50,4)` score | `(50,4)` success |
|---|---:|---:|---:|---:|
| baseline DDPM | 0.114 | 0.00 | 0.696 | 0.25 |
| geom alpha 0.50 | 0.124 | 0.00 | 0.214 | 0.00 |
| geom alpha 0.75 | 0.280 | 0.00 | 0.300 | 0.00 |
| geom alpha 1.00 | 0.454 | 0.00 | 0.446 | 0.00 |

判断：

- 几何 fallback 可以提高最坏低-denoising `(25,2)` 的平均 coverage，但没有带来任何 success。
- 它严重破坏 `(50,4)`，说明 hand-coded geometry 与 learned policy 的 contact behavior 冲突。
- 不能作为 final project 主贡献。

### 6. Action-space clipping

结论：不能作为主贡献。

想法：

- 检查 low-denoising failure 是否只是 final action target 没有显式投影到 Push-T action space `[0,512]^2`。

4-seed DDPM smoke:

| method | `(25,2)` score | `(25,2)` success | `(50,4)` score | `(50,4)` success |
|---|---:|---:|---:|---:|
| clip | 0.113 | 0.00 | 0.667 | 0.00 |

判断：

- clipping 没有救回低-denoising setting。
- 这是标准 deployment hygiene，不是创新。

## Evaluation Caution Found During Search

DDPM 是 stochastic sampler。当前 official runner 匹配的是 Push-T environment seed，但没有严格匹配 policy sampling noise。因此：

- DDIM deterministic comparisons 比较稳定；
- DDPM postprocess smoke tests 只能作为粗略探针，不能用 4-seed 差异作正式 claim；
- 若未来要正式比较 stochastic DDPM variants，必须控制 `torch.Generator` 或固定 policy sampling seed，并做更多 seeds。

为此，`scripts/run_official_kh_grid.py` 已增加 `--policy-seed`。当设置该参数时，wrapper 会在每次 `predict_action` 前重置 `torch` 和 `cuda` RNG，使不同 postprocess variants 使用相同的 diffusion sampling noise 序列。

CRN 4-seed check with `--policy-seed 12345`:

| method | `(25,2)` score | `(25,2)` success | `(50,4)` score | `(50,4)` success |
|---|---:|---:|---:|---:|
| baseline DDPM | 0.186 | 0.00 | 0.735 | 0.75 |
| clip | 0.186 | 0.00 | 0.735 | 0.75 |
| rate-limit | 0.154 | 0.00 | 0.736 | 0.75 |
| geometric fallback | 0.393 | 0.00 | 0.456 | 0.00 |

CRN 判断：

- action-space clipping 确认无效；
- rate-limit 不是可靠正向改进；
- geometric fallback 只提高 `(25,2)` coverage，但没有 success，并且破坏 `(50,4)`；
- 因此这些 postprocess variants 不能作为 final contribution。

## Current State

目前找到一个比前面方向更有希望、且已经通过 20-seed official-checkpoint 验证的新主线：**mask-aware temporal keypoint imputation for Push-T low-dimensional occlusion robustness**。

## Current Candidate: Mask-Aware Temporal Keypoint Imputation

### Motivation

Push-T low-dimensional observation 包含 9 个 block keypoints、agent position，以及对应 `obs_mask`。官方 lowdim Diffusion Policy 的 `predict_action` 实际只使用 `obs_dict["obs"]`，没有使用 `obs_mask`。如果真实部署中 keypoint sensor/estimator 遮挡并把不可见 keypoints 置零，policy 会把缺失值当作真实坐标，导致严重失效。

### Related-work status

已检索：

- `Diffusion Policy keypoint_visible_rate imputation`
- `Diffusion Policy obs_mask keypoint`
- `Push-T keypoint occlusion diffusion policy imputation`
- `robot imitation keypoint occlusion temporal imputation policy`
- `visual keypoint occlusion imputation robot manipulation policy`

目前没有发现“official Diffusion Policy Push-T lowdim checkpoint + masked keypoint zeroing + test-time temporal carry-forward imputation”的直接相同工作。相关大类包括 robust perception、occlusion handling、state estimation 和 temporal filtering，因此 claim 必须写窄：

```text
We propose a lightweight test-time observation wrapper for low-dimensional Push-T keypoint occlusion, not a general visual Diffusion Policy method.
```

### Implementation

`scripts/run_official_kh_grid.py` 新增：

- `--keypoint-visible-rate`
- `--occlude-masked-keypoints`
- `--obs-impute {none,carry_forward}`

关键逻辑：

- 当 `--occlude-masked-keypoints` 开启时，对 `obs_mask=false` 的 18 维 block keypoint 坐标置零，模拟真实 keypoint dropout。
- `carry_forward` 使用同一 rollout 内最近一次可见 keypoint 坐标填补缺失值。
- agent position 不改。
- policy weights、sampler、horizon 不改。

### 4-seed pilot on official checkpoint

Sampler: DDIM, `(k,h)=(100,8)`, official checkpoint, 4 test seeds.

| visible rate | no imputation score | no imputation success | carry-forward score | carry-forward success |
|---:|---:|---:|---:|---:|
| 0.50 | 0.119 | 0.00 | 0.999 | 1.00 |
| 0.25 | 0.113 | 0.00 | 0.521 | 0.25 |

### Interpretation

This is the first candidate in this search with a strong positive official-checkpoint signal:

- under 50% keypoint visibility, carry-forward imputation nearly restores clean-performance rollout quality;
- under severe 25% visibility, it still improves score and success but does not fully solve the task;
- the claim is about robustness to masked lowdim keypoint dropout, not about improving clean full-observation performance.

### Required next validation before using as final headline

已完成 formal 20-seed evaluation:

```bash
for visible_rate in 0.5 0.25; do
  for impute in none carry_forward; do
    CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache \
      /root/FinalProject/.venv_official/bin/python scripts/run_official_kh_grid.py \
      --sampler ddim \
      --obs-impute "$impute" \
      --keypoint-visible-rate "$visible_rate" \
      --occlude-masked-keypoints \
      --output-root "official_reproduction/pusht_official_occluded_impute_n20/vr_${visible_rate}/${impute}" \
      --csv-out "results/official_occluded_impute_n20_vr_${visible_rate}_${impute}.csv" \
      --n-test 20 --n-envs 5 --pairs 100,8
  done
done
```

Formal 20-seed paired analysis:

| visible rate | zero-masked score | zero-masked success | carry-forward score | carry-forward success | delta | 95% bootstrap CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.177 | 0.00 | 0.901 | 0.75 | +0.725 | [+0.585, +0.844] |
| 0.25 | 0.134 | 0.00 | 0.575 | 0.20 | +0.440 | [+0.294, +0.587] |

Generated artifacts:

- `results/official_occluded_imputation_seed_scores.csv`
- `results/official_occluded_imputation_summary.csv`
- `figures/official_occluded_imputation.png`
- `scripts/analyze_occluded_imputation.py`

Decision:

```text
This is now the strongest candidate for the final project contribution.
```

Safe claim:

```text
For the official low-dimensional Push-T checkpoint under synthetic masked-keypoint dropout,
a lightweight test-time carry-forward imputation wrapper substantially improves robustness
without retraining the policy. The claim is specific to low-dimensional keypoint observations
and should not be presented as a general visual Diffusion Policy algorithm.
```

## Claude Review Follow-up: Baseline and Source Audit

Claude 的 review 基本成立，并且已经按其建议补实验。

### Source-level verification

官方源码核实到三点：

- `diffusion_unet_lowdim_policy.py:99-108` 的 `predict_action` 只 assert/use `obs_dict["obs"]`，通过 `self.normalizer["obs"].normalize(obs_dict["obs"])` 进入 policy，没有读取 `obs_mask`。
- `pusht_keypoints_runner.py:204-209` 会把环境 observation 拆成 `obs` 和 `obs_mask` 两部分传给 policy。
- `pusht_keypoints_env.py:91-122` 内置 `keypoint_visible_rate` 并生成 mask，但环境返回的 `obs = kps.flatten()` 仍是真实 keypoint 坐标；被置零的 `vis_kps` 只用于 rendering。因此我们的 `--occlude-masked-keypoints` 明确是在模拟真实 keypoint detector 遮挡时“不返回坐标/返回零”的部署输入，而不是官方 lowdim env 默认就会让 policy 看到零坐标。
- `train_diffusion_unet_lowdim_workspace.yaml:19` 默认 `keypoint_visible_rate: 1.0`，说明官方 lowdim checkpoint 是按 full visibility 配置训练/复现的。

这要求正式报告必须精确表述：

```text
The official environment exposes visibility masks, but the low-dimensional policy ignores them.
Because the official low-dimensional observation still contains ground-truth keypoint coordinates
even when a keypoint is marked invisible, we simulate detector-style occlusion by zeroing masked
keypoint coordinates before policy inference, then compare mask-aware imputation strategies.
```

### Strawman baseline issue

Claude 指出 zero-fill baseline 过弱，这个批评成立。我们补了四个对照：

- `mean_prior`: 用 checkpoint normalizer 中的训练集 keypoint 均值填补；
- `frame_hold`: 使用最近一次 18 个 keypoint 坐标全部可见的整帧快照；
- `carry_forward`: 对每个 keypoint 坐标独立使用最近一次可见值；
- `linear`: 对每个 keypoint 坐标独立做一阶线性外推，外推结果 clamp 到 `[0,512]`；
- `oracle`: 不可部署上界，保留真实坐标。

实现位置：

- `scripts/run_official_kh_grid.py`
- `scripts/analyze_occluded_imputation.py`

正式评估设置：

- official Push-T lowdim checkpoint；
- DDIM sampler, `(k,h)=(100,8)`；
- 20 matched test seeds；
- visible rate: 50% and 25%；
- paired bootstrap CI；
- policy weights / sampler / horizon 不改，只做 inference-time observation preprocessing。

### Formal 20-seed results with imputation baselines

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

Artifacts:

- `results/official_occluded_imputation_all_summary.csv`
- `results/official_occluded_imputation_all_seed_scores.csv`
- `results/official_occluded_imputation_all_seed_scores_wide.csv`
- `figures/official_occluded_imputation_all.png`

### Updated interpretation

原先把 headline 写成 carry-forward 已经不够准确。正式主张应更新为：

```text
We identify a mask-handling failure mode in the official low-dimensional Push-T Diffusion Policy:
the runner provides keypoint visibility masks, but the policy ignores them. Under detector-style
masked-keypoint dropout, simple mask-aware temporal imputation at test time substantially restores
official-checkpoint performance without retraining. Per-keypoint temporal imputation
(carry-forward or linear extrapolation) is much stronger than zero-fill, training-mean fill, and
full-frame hold baselines.
```

Metric-specific conclusion:

- 50% visible: carry-forward has slightly higher mean score than linear (0.901 vs 0.875), while linear has higher success (0.85 vs 0.75). Their paired score delta vs each other is small and CI crosses zero, so we should not claim one strictly dominates the other.
- 25% visible: linear has higher mean score and success than carry-forward (0.668/0.35 vs 0.575/0.20), but paired score delta CI still crosses zero; write this as a trend, not a proven dominance claim.
- Oracle remains much higher at 25% visible, so severe occlusion is not solved. The honest claim is partial robustness recovery, not complete occlusion robustness.

Final safe claim:

```text
The contribution is a lightweight, training-free, mask-aware observation wrapper for low-dimensional
Push-T deployment under keypoint dropout. It exposes a failure mode in the official checkpoint and
shows that per-keypoint temporal imputation substantially improves robustness over zero/mean/frame
baselines on matched official rollouts. It is not a general visual Diffusion Policy algorithm and
does not claim improvement under clean full-observation rollouts.
```

### Related-work re-check on 2026-06-11

检索关键词：

- `"Diffusion Policy" "obs_mask" keypoint imputation`
- `"Push-T" keypoint occlusion imputation "Diffusion Policy"`
- `"keypoint_visible_rate" "Diffusion Policy" imputation`
- `"PushTKeypointsEnv" "obs_mask"`
- `"carry-forward" "Diffusion Policy" keypoint`
- `"linear extrapolation" "Diffusion Policy" keypoint`
- GitHub domain-restricted variants of the above queries

检索结果：

- 官方 Diffusion Policy 论文确认 Push-T 有 RGB 与 9 个 2D keypoint 两种 observation variants，并且 keypoints 来自 T block ground-truth pose；这支持我们研究的是官方 lowdim keypoint setting，而不是另造任务。
- 官方 repo / config 确认公开了 experiment logs/checkpoints/configs，并且 Push-T lowdim config 中 `keypoint_visible_rate: 1.0`。
- 找到 LeRobot 的 PushT keypoints Diffusion Policy model card，但没有看到针对 `obs_mask` 或 keypoint occlusion imputation 的评估。
- 找到通用 missing data / skeleton keypoint imputation 工作，例如 diffusion-based imputation、DISK skeleton imputation。这些是相关背景，不是同一 robotics policy deployment setting。
- 没有发现“official Diffusion Policy Push-T lowdim checkpoint + masked keypoint zeroing + no-retraining temporal imputation wrapper + matched rollout evaluation”的直接相同工作。

外部 sources:

- Diffusion Policy paper: https://arxiv.org/html/2303.04137v5
- Official Diffusion Policy repo: https://github.com/real-stanford/diffusion_policy
- Official lowdim Push-T config example: https://diffusion-policy.cs.columbia.edu/data/experiments///low_dim/pusht/diffusion_policy_transformer/config.yaml
- LeRobot PushT keypoints model card: https://huggingface.co/lerobot/diffusion_pusht_keypoints
- General skeleton/keypoint imputation background: https://pmc.ncbi.nlm.nih.gov/articles/PMC12791013/
- General diffusion missing-data imputation background: https://openreview.net/pdf?id=QUANtQnx30l

Risk after re-check:

- 不能声称“我们发明 temporal imputation”或“我们解决遮挡鲁棒性”。
- 可以声称“we identify and evaluate a specific mask-handling failure mode in the official lowdim Push-T Diffusion Policy deployment path, and show that a training-free mask-aware temporal observation wrapper substantially improves robustness in this setting.”
