# Oral Notes

建议中文讲：项目最终主线是复现官方 Diffusion Policy Push-T，并实现 DDIM sampler replacement。不要再讲 scheduler 是主贡献。核心结果是官方 matched seeds 下 `(25,2)` 从 0.155 到 0.900，`(50,4)` 从 0.653 到 0.929；但 `(100,8)` 下 DDPM 仍然更强，所以 claim 是 DDIM 改善低/中 denoising frontier，不是全面打败 DDPM。
