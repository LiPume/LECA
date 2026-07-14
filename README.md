# LECA 螺栓检测：论文版本保护与机制审计

本仓库保留已发表 LECA 的本地代码快照，并补充可复现实验、结构审计和面向答辩的机制证据。主源码位于 `ultralytics-main/`，Conda 环境名为 `yolo`，完整环境见 `environment.yml`。

- 历史训练入口位于 `ultralytics-main/train_*.py`；受控机制审计入口位于 `tools/train_mechanism_smoke.py`。
- 数据路径通过本地数据集 YAML 配置。数据集、标注、图片、权重、训练输出和可视化均不进入 Git。
- 标签 `paper-original` 固定保存审计前的已发表版本；`main` 不用于改写论文实现。
- 当前受控机制验证位于 `exp/mechanism-audit`，采用拓扑匹配的 Baseline/ECA/LECA；这不是论文复现。
- 本地运行结果保存于被忽略的 `runs_repro/`，中间特征图保存于被忽略的 `artifacts/visualizations/`。

长时间训练前，请先阅读 [代码审计](docs/CODE_AUDIT.md) 和 [复现实验协议](docs/REPRODUCTION_PROTOCOL.md)。
