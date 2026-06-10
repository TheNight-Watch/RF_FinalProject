# Final Project Plan

## Project Title

**Joint Compute-Control Budgeting for Receding-Horizon Diffusion Policy**

Subtitle:

**Iso-Compute Allocation of Denoising and Replanning for Efficient Robotic Manipulation**

## One-Sentence Pitch

This project reproduces Diffusion Policy and studies how to jointly allocate a fixed inference budget between denoising quality and replanning frequency, showing whether a budget-aware joint scheduler can improve the efficiency-stability tradeoff over fixed or single-axis baselines.

## Project Fit

This project is designed for the Advanced Topics for Robotics final project requirement:

- Reproduce an existing robotics paper.
- Implement a meaningful improvement on top of it.
- Avoid large-scale training and heavy compute.
- Produce English proposal, presentation, and final report in a CoRL-style format.
- Be feasible for a coding agent to complete end-to-end with limited human intervention.

The base work is **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**, RSS 2023.

Project page: https://diffusion-policy.cs.columbia.edu/

## Core Research Question

Diffusion Policy uses iterative denoising to predict action chunks and executes them in a receding-horizon manner. At deployment time, two resources compete:

- **Denoising budget**: how many denoising steps / NFEs to spend per policy call.
- **Replanning budget**: how frequently the robot should replan instead of continuing the current action chunk.

Let:

- `k` = number of denoising steps / NFEs per policy call.
- `h` = number of executed control steps before the next replan.

The amortized compute per control step can be approximated as:

```text
B ~= k / h
```

A more realistic form is:

```text
B = (C_obs + k * C_denoise + C_sched) / h
```

where `C_obs` is observation encoding cost, `C_denoise` is one denoising step cost, and `C_sched` is scheduler overhead.

The main research question is:

> Under a fixed per-step inference budget `B`, should compute be spent on more accurate denoising (`larger k`) or more frequent replanning (`smaller h`)? Can a state-dependent joint scheduler outperform the best fixed `(k, h)` and single-axis adaptive baselines?

## Novelty Boundary

The proposal must avoid claiming that adaptive denoising, adaptive chunking, or uncertainty-based failure detection alone are new.

Related work already covers important single-axis pieces:

- **AAC, 2026**: adjusts action chunk size / replanning frequency, with fixed denoising.
- **DVAC, 2026**: uses denoising variance to decide when to replan, primarily adjusting execution horizon.
- **D3P, 2025**: dynamically allocates denoising steps, but uses a learned RL adaptor.
- **RTI-DP, 2025**: accelerates Diffusion Policy inference with training-free fast denoising.
- **Two-Steps Diffusion Policy / Genetic Denoising, 2025**: studies extremely low-NFE diffusion policy.
- **FIPER / FAIL-Detect / Diff-DAgger**: use uncertainty, diffusion loss, entropy, or disagreement for failure prediction.
- **TIDAL, 2026**: studies compute redistribution for high-frequency VLA control.

This project's defensible novelty is:

> Existing methods mostly optimize one axis: denoising steps or replanning frequency. This project formulates and evaluates the joint allocation of denoising and replanning under an iso-compute constraint, then proposes a training-free scheduler that chooses `(k, h)` on the same compute frontier.

## Expected Contribution

The project should have two layers:

1. **Analysis contribution**
   - Construct an iso-compute grid over `(k, h)`.
   - Characterize when denoising quality matters more and when frequent replanning matters more.
   - Visualize score / success / latency / smoothness across equal-compute frontiers.

2. **Method contribution**
   - Implement a training-free joint scheduler that selects `(k, h)` under a fixed compute budget.
   - Compare it against fixed baselines and single-axis adaptive baselines.

## Target Difficulty

Expected difficulty: **4.0-4.2 / 5**.

This is harder than a simple reproduction but avoids large-scale VLA or humanoid training. It should be feasible because the main implementation is in inference-time runners and evaluation scripts rather than full model training.

## Compute Requirements

Minimum:

- CPU can run small Push-T evaluations with official checkpoints, but slowly.

Recommended:

- One consumer GPU with 8-16GB VRAM.
- Use official Diffusion Policy checkpoint whenever possible.
- Avoid full large-scale retraining.

Not required:

- Multi-GPU training.
- A800/4090-class resources.
- Real robot hardware.
- Large-scale VLA model training.

## Main Environment

Primary:

- **Push-T** online evaluation from Diffusion Policy.

Why:

- Standard Diffusion Policy benchmark.
- Closed-loop rollout available.
- Contact-rich pushing creates a real reactivity vs denoising-quality tradeoff.
- CPU fallback is possible.

Supplementary options:

- **Robomimic Can or Square**
  - More precision-sensitive.
  - Useful if setup is stable.
- **M7 offline evaluation from HW2**
  - Reuses existing work.
  - Good supplementary evidence, but not enough alone because the key contribution is closed-loop replanning.

## Experimental Variables

Use:

```text
k in {2, 4, 8, 16}
h in {1, 2, 4, 8}
B = k / h
```

Primary iso-compute frontiers:

```text
B = 1: (2,2), (4,4), (8,8)
B = 2: (2,1), (4,2), (8,4), (16,8)
B = 4: (4,1), (8,2), (16,4)
```

If runtime is too high, reduce to:

```text
k in {2, 4, 8}
h in {1, 2, 4}
```

and run more seeds for fewer grid points.

## Metrics

Must collect:

- Task score / success rate.
- Average `k` per policy call.
- Average executed horizon `h`.
- Average NFE per episode.
- Amortized compute `k / h`.
- Policy calls per episode.
- Wall-clock rollout time.
- Chunk-boundary action discontinuity.
- Trajectory smoothness, e.g. velocity variance or jerk proxy.

Recommended:

- Phase-wise score around contact vs free-space motion.
- Failure case categories.
- Rollout videos for representative runs.

## Baselines

The project must compare against strong baselines, not only the default Diffusion Policy.

Required baselines:

1. **Default Diffusion Policy**
   - Original or standard `k`, standard execution horizon.

2. **Best fixed `(k, h)` under the same budget**
   - This is the most important baseline.
   - The adaptive method must be compared against a tuned fixed baseline.

3. **Denoising-only baseline**
   - Vary `k`, keep `h` fixed.

4. **Replanning-only baseline**
   - Keep `k` fixed, vary `h`.

5. **Heuristic AAC/DVAC-style baseline**
   - Use uncertainty or action entropy to adjust `h`, while keeping `k` fixed.

Proposed method:

6. **Joint iso-compute scheduler**
   - Selects `(k, h)` from a fixed-budget candidate set.

## Proposed Scheduler

The first version should be simple and training-free.

Input features can include:

- Denoising convergence signal.
- Multi-sample action disagreement.
- Predicted action chunk smoothness.
- Chunk-boundary discontinuity.
- Recent progress toward task goal.
- Push-T-specific phase features:
  - pusher-object distance.
  - object-goal distance.
  - contact state proxy.

Candidate actions:

```text
S_B = {(k, h) | k / h = B}
```

Example for `B = 2`:

```text
S_2 = {(2,1), (4,2), (8,4), (16,8)}
```

Scheduler behavior:

- High-risk / contact / unstable action prediction:
  - prefer smaller `h`, more frequent replanning.
- Precision-sensitive but stable state:
  - prefer larger `k`, longer chunk.
- Easy free-space state:
  - prefer cheaper candidate if allowed by the budget set.

Implementation recommendation:

Start with a rule-based scheduler calibrated on validation episodes. Do not train a neural adaptor unless all core experiments are already complete.

## Implementation Plan

### Phase 0: Repository Setup

- Clone or copy the official Diffusion Policy repository.
- Create a reproducible environment.
- Record exact Python, PyTorch, CUDA, and package versions.
- Add a local project folder for scripts, logs, generated plots, and reports.

Deliverable:

- `README.md` with setup instructions.
- Environment log.

### Phase 1: Baseline Reproduction

- Run the official Push-T checkpoint evaluation.
- Confirm online rollout works.
- Save at least one rollout video.
- Record score, runtime, and policy-call count.
- If GPU is available, run a short training or checkpoint evaluation to verify full pipeline.

Deliverable:

- Baseline evaluation JSON.
- Baseline rollout video.
- Short reproduction section for report.

### Phase 2: Configurable Inference Runner

Modify or wrap the evaluation runner so it supports:

- Custom denoising steps `k`.
- Custom execution horizon `h`.
- Logging of policy calls, denoising calls, runtime, and action chunks.
- Fixed random seeds.
- Batched evaluation over multiple episodes.

Important:

- Keep implementation minimally invasive.
- Preserve original behavior as a baseline mode.

Deliverable:

- `eval_kh_grid.py` or equivalent.
- Unit/smoke test for one `(k, h)` pair.

### Phase 3: Static Iso-Compute Grid

- Run the `(k, h)` grid.
- Evaluate several seeds per point if runtime allows.
- Save all raw logs.
- Generate:
  - Heatmap over `(k, h)`.
  - Iso-compute line plots.
  - Score vs compute Pareto plot.
  - Smoothness vs compute plot.

Deliverable:

- `results/grid_results.csv`.
- `figures/kh_score_heatmap.png`.
- `figures/iso_compute_curves.png`.
- `figures/pareto_score_compute.png`.

### Phase 4: Phase-Wise Analysis

- Segment rollout into task phases.
- For Push-T, useful proxies are:
  - far from object.
  - initial contact.
  - pushing object.
  - final alignment near target.
- Analyze which `(k, h)` works best in each phase.

Deliverable:

- Phase-wise metric table.
- Plot showing preferred `(k, h)` by phase.

### Phase 5: Joint Scheduler

- Implement the training-free scheduler.
- Restrict choices to an iso-compute set `S_B`.
- Start with a rule-based policy.
- Optionally tune thresholds on validation seeds.
- Test on held-out seeds.

Deliverable:

- Scheduler implementation.
- Scheduler config file.
- Validation and test logs.

### Phase 6: Baseline Comparison

Compare:

- Default DP.
- Best fixed `(k,h)`.
- Denoising-only adaptive/fixed baseline.
- Replanning-only adaptive/fixed baseline.
- AAC/DVAC-style heuristic baseline.
- Joint scheduler.

Run all methods under the same average budget `B`.

Deliverable:

- Main comparison table.
- Bar plot of score / success.
- Bar plot of policy calls and NFE.
- Smoothness plot.

### Phase 7: Supplementary Environment

If time allows:

- Run Robomimic Can or Square.
- Otherwise run M7 offline evaluation from HW2.

Goal:

- Show whether the same allocation principle generalizes.
- Do not make supplementary results the core claim unless closed-loop evaluation is available.

Deliverable:

- Supplementary results table.
- Short discussion of generalization.

### Phase 8: Report and Slides

Prepare:

- English final report in CoRL-like structure.
- English presentation slides.
- Optional Chinese oral notes if needed.

Deliverable:

- `report/final_report.pdf`.
- `slides/final_presentation.pptx` or `.pdf`.
- `artifacts/` folder containing videos, plots, logs, and source patches.

## Report Outline

1. **Introduction**
   - Real-time deployment challenge.
   - Denoising quality vs replanning frequency.
   - Summary of contributions.

2. **Related Work**
   - Diffusion Policy.
   - Adaptive action chunking.
   - Fast or adaptive denoising.
   - Failure prediction / uncertainty.
   - Compute-aware robot policy inference.

3. **Problem Formulation**
   - Define `k`, `h`, and `B ~= k / h`.
   - Define iso-compute frontiers.
   - Explain evaluation objective.

4. **Reproduction**
   - Push-T setup.
   - Diffusion Policy baseline reproduction.
   - Environment and checkpoint details.

5. **Static Iso-Compute Analysis**
   - Grid results.
   - Pareto frontier.
   - Interpretation of when denoising or replanning matters.

6. **Method**
   - Joint scheduler.
   - Input features.
   - Candidate set.
   - Training-free calibration.

7. **Experiments**
   - Main comparison.
   - Ablations.
   - Smoothness and latency.
   - Failure cases.

8. **Discussion**
   - Why the best budget allocation changes by phase.
   - When fixed policies are enough.
   - When joint scheduling helps.

9. **Limitations**
   - Limited environments.
   - Heuristic scheduler.
   - Approximate compute model.
   - No real-robot validation.

10. **Conclusion**
    - Main empirical finding.
    - Practical deployment lesson.

## Slide Outline

Recommended 10-slide structure:

1. **Title**
   - Project title, team members, course.

2. **Motivation**
   - Diffusion Policy is strong but inference is expensive.
   - Replanning improves reactivity but costs more policy calls.

3. **Key Question**
   - Spend compute on denoising or replanning?

4. **Formulation**
   - `k`, `h`, `B ~= k / h`.
   - Iso-compute frontier diagram.

5. **Related Work**
   - Single-axis methods vs joint allocation.

6. **Reproduction**
   - Push-T baseline and setup.

7. **Iso-Compute Analysis**
   - Heatmap and Pareto plot.

8. **Joint Scheduler**
   - Rule-based or calibrated scheduler.
   - Candidate set under fixed budget.

9. **Results**
   - Comparison against best fixed and single-axis baselines.

10. **Takeaways**
    - Main finding.
    - Limitations and future work.

## Success Criteria

Minimum successful project:

- Reproduce Diffusion Policy Push-T evaluation.
- Implement configurable `(k, h)` runner.
- Run static iso-compute grid.
- Produce plots and analysis.
- Write report and slides.

Strong successful project:

- Add training-free joint scheduler.
- Beat or match the best fixed `(k, h)` under the same compute on at least one meaningful metric.
- Include strong baseline comparison.
- Include phase-wise analysis and videos.

Excellent project:

- Include supplementary Robomimic or M7 evaluation.
- Show task-dependent allocation differences.
- Provide clean code, reproducibility scripts, report, and slides.

## Risks and Mitigations

### Risk: Adaptive scheduler does not beat best fixed baseline

Mitigation:

- The static iso-compute analysis is still a valid contribution.
- Report the scheduler honestly and emphasize empirical characterization.
- Try phase-based scheduler instead of global risk-only scheduler.

### Risk: Low-NFE methods are always enough

Mitigation:

- Use contact-rich and precision-sensitive tasks.
- Analyze failure cases where low `k` produces mode jumps or unstable chunks.

### Risk: Evaluation runtime is too high

Mitigation:

- Reduce grid size.
- Use fewer seeds initially.
- Use official checkpoint.
- Prioritize `B = 2` frontier first.

### Risk: Robomimic setup is unstable

Mitigation:

- Keep Push-T as the main environment.
- Use M7 offline evaluation as supplementary evidence.

### Risk: Novelty gets challenged

Mitigation:

- Avoid claiming new uncertainty metrics.
- Avoid claiming adaptive chunking or adaptive denoising as the main contribution.
- Emphasize iso-compute joint allocation and best-fixed baseline comparison.

## Recommended Development Order for Claude/Codex

1. Inspect the official Diffusion Policy repository and existing HW2 work.
2. Set up Push-T evaluation with an official checkpoint.
3. Add logging for policy calls, NFE, `k`, `h`, runtime, score, and action chunks.
4. Implement fixed `(k,h)` evaluation.
5. Run a small smoke grid.
6. Generate first heatmap and verify metrics.
7. Scale to the selected iso-compute frontiers.
8. Implement single-axis baselines.
9. Implement joint scheduler.
10. Run final comparisons.
11. Generate final figures.
12. Write proposal/report/slides.

## Naming Convention

Suggested folder layout:

```text
FinalProject/
  PROJECT_PLAN.md
  diffusion_policy/                 # cloned upstream repo or submodule
  scripts/
    eval_kh_grid.py
    run_iso_compute_experiments.sh
    plot_iso_compute.py
  configs/
    kh_grid_push_t.yaml
    scheduler_push_t.yaml
  results/
    raw/
    grid_results.csv
    scheduler_results.csv
  figures/
    kh_score_heatmap.png
    iso_compute_curves.png
    pareto_score_compute.png
  report/
    final_report.tex
    final_report.pdf
  slides/
    final_presentation.pptx
```

## Final Recommendation

Use this as the main final project direction.

The project is valuable because it targets a real deployment tradeoff in Diffusion Policy. It is novel enough for a course project because it frames denoising and replanning as a joint iso-compute allocation problem rather than another single-axis adaptive method. Its difficulty is appropriate because it requires serious robotics evaluation and analysis while avoiding large-scale model training.
