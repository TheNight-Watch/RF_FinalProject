# Oral Notes

English or Chinese can be used for the oral presentation. Suggested short Chinese framing:

本项目复现并分析 Diffusion Policy 的推理时结构：每次策略调用需要若干 denoising steps，并在 receding horizon 中执行一段 action chunk。官方源码通过 GitHub archive 获取，官方 Push-T checkpoint 已下载并跑通 50 个 test seed；当前 Python 3.12 / Gym 0.26 环境需要兼容补丁，所以实测分数低于 checkpoint 文件名中的 0.969。优化版增加了官方 checkpoint 的固定 `(k,h)` frontier：结果显示官方模型在低 denoising step 下性能明显下降，说明不能把 surrogate 结论过度推广。核心改进问题是在固定预算下，计算量应该用于更高质量的 denoising，还是更频繁地 replanning。20 个 matched seeds 的 surrogate 实验显示，score-oriented scheduler 提高均分并降低 policy calls/动作抖动，safe scheduler 则提高 success，这形成了更清晰的 Pareto tradeoff。
