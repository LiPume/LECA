# 受控实验与复现协议

## 不可违反的边界

- `paper-original` 是不可变论文代码快照；`main` 不覆盖论文实现。任何新公式只能从 `exp/leca-v2` 开始。
- 不修改数据划分、不筛选测试样本、不覆盖权重、不删除原实验输出、不强推 Git，也不上传数据、图片、权重、runs、artifacts 或可视化。
- 每条执行命令记录于 `docs/COMMAND_LOG.md`；Git 中只保留小型汇总表、代码、配置、脚本和文稿。
- 每个判断注明：**已确认（Confirmed）**为代码/实验直接支持，**已观察（Observed）**为实验现象但未证因果，**假设（Hypothesis）**为待检验解释。

## 当前实验对象

当前 `exp/mechanism-audit` 是**受控机制审计**，不是论文的完全复现：它在不动论文快照的前提下，以 8 个相同位置构建 Identity Baseline、ECA 和 LECA。历史训练/验证划分的重复问题保留并明确标记，因此不用其验证指标作为性能证据。Hard Test 仅用于训练完成后的评估；不用于早停、调参或选权重。

## 每次长训练前的门槛

1. 保存当前分支、模型 YAML、数据 YAML、预训练权重哈希与实际环境信息。
2. 输出实例化图的模块清单、参数量、FLOPs 工具和输入尺寸，确认三种模型仅在目标注意力模块不同。
3. 固定 seed、deterministic、优化器、学习率、权重衰减、batch、imgsz、epochs、patience、增强、AMP、workers、cache 和设备。
4. 先执行单 batch 前向/反向与有限值检查，再执行 1 epoch smoke test；确认损失、标量梯度、权重保存、验证和评估脚本均正常。
5. 若出现阻塞错误，记录错误、环境和命令后停止；不静默绕过。

## 性能测量口径

- 参数量：对确切实例化图计算 `sum(p.numel() for p in model.parameters())`。
- FLOPs：同一工具/版本、FP32、输入 `[1,3,640,640]`，记录单位。
- 延迟/FPS：同 GPU、batch=1、640、相同精度、固定 warm-up 和计时次数；说明是否包含预处理、NMS、数据传输。
- 测试：由验证集选择 `best.pt` 后，对固定 Hard Test 运行一次。一次 seed 的结果只称观察；三种子后才报告均值和标准差。

## 中间统计与图像证据

LECA hook 默认关闭。开启后按 epoch、按层汇总 `alpha,beta,gamma,mu,var,corr,w_eca,w_sup,w_rec,w_bri,w_final` 的 mean/std/p05/p25/p50/p75/p95/min/max/NaN/Inf；原始张量仅放在忽略目录 `artifacts/statistics/`。特征图、叠加图和压力测试图片只放在 `artifacts/visualizations/`；Git 只保留不含图像内容的汇总 CSV/JSON。
