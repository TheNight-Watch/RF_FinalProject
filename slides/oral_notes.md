# Oral Notes

English or Chinese can be used for the oral presentation. Suggested short Chinese framing:

本项目复现并分析 Diffusion Policy 的推理时结构：每次策略调用需要若干 denoising steps，并在 receding horizon 中执行一段 action chunk。官方源码通过 GitHub archive 获取，官方 Push-T checkpoint 已下载并跑通 50 个 test seed；当前 Python 3.12 / Gym 0.26 环境需要兼容补丁，所以实测分数低于 checkpoint 文件名中的 0.969。核心改进问题是在固定预算下，计算量应该用于更高质量的 denoising，还是更频繁地 replanning。实验显示，在接触阶段高频 replanning 更重要，而在 free-space 和最终对齐阶段可以使用更长 chunk 和更多 denoising。联合调度器在相同平均预算下提升了 score，同时减少了 policy calls 和动作抖动。
