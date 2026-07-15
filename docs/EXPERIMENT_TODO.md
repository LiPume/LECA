# LECA 实验待办

状态：`[x]` 已完成；`[~]` 已完成但证据边界有限；`[ ]` 待完成。所有输出目录均在 `.gitignore` 中；不得修改 `paper-original`、历史数据、权重或运行记录。

## P0：审计与受控验证基础

- [x] 记录环境、源码位置、训练入口和 Git 保护策略。
- [x] 逐项核对 LECA 公式、形状、参数注册、梯度和数值风险。
- [x] 审计数据划分：历史 train/val 有 5 组精确重复。
- [x] 审计模型公平性：历史 `yolo11.yaml` 含 8 个隐式 LECA，不能称为 Baseline。
- [x] 建立不改论文版本的拓扑匹配 Baseline/ECA/LECA，确认参数量 2,624,080/2,624,116/2,624,140。
- [x] 单 batch 前向、反向、NaN/Inf 检查和三个模型各 1 epoch smoke test。
- [x] 默认关闭的 LECA 聚合 hooks；统计仅落地至本地 `artifacts/statistics/`。
- [x] 选定 Hard Test 作为性能评估集；其与 `trainDataV3` 无 SHA256 精确重复。
- [ ] 对 Hard Test 与训练数据做 pHash/连续帧近重复审计，并写入仅汇总的报告。
- [ ] 固化每次训练的预训练权重哈希、配置哈希、实际 GPU、CUDA、PyTorch、Ultralytics 版本。

## P1：受控性能与模块价值

- [~] seed=42 的 Baseline/ECA/LECA Hard Test 已完成；结果只可称一次受控观察。
- [ ] 在完全相同训练配置下补 seed=123、2026，并报告均值 ± 标准差。
- [ ] 用同一工具、同一 640 输入统计 Params、FLOPs、batch=1 延迟和 FPS；明确是否含预处理、NMS 和数据传输。
- [x] seed=42 独立重训练八种组合：ECA、ECA+Var、ECA+Rec、ECA+Bri、ECA+Var+Rec、ECA+Var+Bri、ECA+Rec+Bri、ECA+Var+Rec+Bri；已统一评估 Hard Test。
- [x] seed=42 阶段级插入位置消融：无 LECA、Backbone 4 处、融合路径 4 处、P3/P4/P5 3 处、Full 8 处；中间三组均先 smoke 再独立重训练，Full 四项指标最高。
- [ ] 对关键组合补三种子。所有模型固定训练数据、预训练权重、早停、增强和评估脚本；Hard Test 不参与早停、调参或选权重。
- [x] 推理时 beta/alpha/gamma 置零已单独标为敏感性分析；没有替代上述重训练消融。

## P2：答辩机制证据

- [x] 已输出 Hard Test 代表样本中 `model.16.eca` 的 ECA/LECA 对照特征图与权重表，保存在 `artifacts/visualizations/hard_feature_maps/`。
- [ ] H1：搜索均值接近、方差不同的真实通道；可视化空间响应并对比权重，不给通道强行命名。
- [ ] H2：按 TP/FP/FN 和反光、孔洞、铆钉、暗光类型统计 variance、`w_sup`、最终权重；报告 Spearman、效应量和分布。
- [ ] H3：按低照度/正常样本及 TP/FP/FN 比较 `mu`、`w_rec`；推理时 alpha=0 只作为诊断。
- [ ] H4：关联 `corr` 与图像灰度均值，并比较不同深度及空间不均匀样本；不得将其直接称为物理亮度。
- [~] 已建立 146 张 `metadata/hard_case_index.csv` AI 单人原图初标并完成分组指标；仍需现场人员复核，且连续场景数不足。
- [x] 已在 146 张 Hard Test 上汇总 8 层 alpha/beta/gamma、`w_sup/w_rec/w_bri/w_stat/w_final` 的均值、分位数与极值，并输出本地逐层曲线。
- [x] 已对同一收敛 Full 模型执行 Var/Rec/Bri 推理中性化，明确标为局部敏感性而非重训练消融。
- [ ] 补充 TP/FP/FN 分布和分支关闭逐图预测对比；图片仅保存本地。

## P3：Controlled Stress Tests

- [ ] 对 Hard Test 本地副本仅施加光度变化：全局明暗、Gamma、径向/线性照度梯度、局部高亮/过曝和对比度下降；标注框不变。
- [ ] 记录每个扰动参数，对 Baseline/ECA/LECA 报告 mAP、Recall、FP、FN 及相对原图下降。
- [ ] 结论使用“受控压力测试观察”，不声称合成扰动等同于真实物理场景。
