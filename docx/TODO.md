# 答辩实验待办

已完成：论文代码审计、拓扑一致的 Baseline/ECA/LECA 构建、三个模型 1 epoch smoke、Hard Test 的 seed=42 受控评估，以及本地 ECA/LECA 特征图证据输出。

接下来按优先级执行：

1. 用相同配置补 Hard Test 的 seed=123、2026，报告均值和标准差；Hard Test 绝不用于早停或调参。
2. 重训练 Var/Rec/Bri 的八组消融，明确“重训练消融”和“推理关闭敏感性”的区别。
3. 给 Hard Test 建立 hard-case 索引，统计 TP/FP/FN 中的方差、权重和误检类型。
4. 完成只改变光度的 Controlled Stress Tests，并保留扰动参数。
5. 输出分层标量曲线、分布图和少量代表特征图；所有图片只留在本地 `artifacts/visualizations/`。

答辩结论必须使用“已确认 / 已观察 / 假设”，不把高方差说成反光、不把低均值说成螺栓、不把深层特征均值说成物理亮度。详细方案见 [`docs/EXPERIMENT_TODO.md`](../docs/EXPERIMENT_TODO.md)。
