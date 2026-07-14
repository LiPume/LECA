# 答辩结果摘要

- Confirmed：当前 LECA 公式主体与设计式匹配，合成前向/反向均有限值。
- Confirmed：当前 `yolo11.yaml` 已含 8 个隐式 LECA，不能称为 Baseline。
- Confirmed：当前注意力对比 YAML 改变了主干结构，参数/FLOPs 对比不公平。
- Confirmed：历史训练/验证划分有 5 组精确重复图像跨集合。
- Not run：新的 smoke、复现、消融、三种子、压力测试。因此没有新的性能数字可用于论文或答辩结论。

补充：受控机制审计分支已经完成 Identity、ECA、LECA 的 1 epoch smoke，并记录了 8 层 LECA 的 alpha/beta/gamma 与权重统计；没有 NaN/Inf。它只说明模块可训练且数值稳定，不能作为最终性能或因果结论。

分支关闭敏感性在 1 epoch 后所有条件均为零检测，说明模型尚未收敛，暂时不能量化 Var/Rec/Bri 的贡献；不能把它解释为“分支没用”。

正式结果边界见 [`docs/RESULTS_SUMMARY.md`](../docs/RESULTS_SUMMARY.md)；3 分钟答辩稿和追问见 [`docs/INTERVIEW_EVIDENCE.md`](../docs/INTERVIEW_EVIDENCE.md)。
