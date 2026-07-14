# 答辩结果摘要

- Confirmed：当前 LECA 公式主体与设计式匹配，合成前向/反向均有限值。
- Confirmed：当前 `yolo11.yaml` 已含 8 个隐式 LECA，不能称为 Baseline。
- Confirmed：当前注意力对比 YAML 改变了主干结构，参数/FLOPs 对比不公平。
- Confirmed：历史训练/验证划分有 5 组精确重复图像跨集合。
- Not run：新的 smoke、复现、消融、三种子、压力测试。因此没有新的性能数字可用于论文或答辩结论。

正式结果边界见 [`docs/RESULTS_SUMMARY.md`](../docs/RESULTS_SUMMARY.md)；3 分钟答辩稿和追问见 [`docs/INTERVIEW_EVIDENCE.md`](../docs/INTERVIEW_EVIDENCE.md)。
