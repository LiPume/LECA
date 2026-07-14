# 答辩实验待办

当前阶段为“保守审计”，不运行全量训练。

1. 恢复论文时的精确源码版本与 YOLO11 Baseline/ECA/LECA YAML。
2. 建立不含重复图像的新分组划分；保留旧划分，不覆盖。
3. 核对三种模型拓扑、模块数、参数量、FLOPs 后，再做单 batch 和 1 epoch smoke。
4. 再依次完成 seed=42、三随机种子、八组分支消融、机制统计和 Controlled Stress Tests。
5. 结论严格使用 Confirmed / Observed / Hypothesis，不把特征统计直接说成反光、螺栓或物理亮度。

完整可执行清单在 [`docs/EXPERIMENT_TODO.md`](../docs/EXPERIMENT_TODO.md)。
