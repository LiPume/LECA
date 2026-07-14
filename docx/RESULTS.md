# 答辩结果摘要

## 可以直接讲的结果

受控分支在同一拓扑、同一 8 个插入位置下比较了 Baseline、ECA、LECA。参数量分别为 2,624,080、2,624,116、2,624,140；三者均通过 seed=42 的 1 epoch smoke test，LECA 8 层统计没有 NaN/Inf，alpha/beta/gamma 都参与更新。

在 Hard Test 的一次受控 seed=42 评估中：

| 模型 | P | R | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | .9142 | .8452 | .9489 | .5848 |
| ECA | .9082 | .8937 | .9573 | .5831 |
| LECA | **.9742** | **.9226** | **.9796** | **.6140** |

这支持“当前 Hard Test、当前 seed 下 LECA 表现更好”的**观察**。Hard Test 和 `trainDataV3` 无 SHA256 精确重复，但尚未完成跨种子、近重复和完整重训练消融，因此不能说成论文的完整复现或因果证明。

中间证据已输出到本地 `artifacts/visualizations/hard_feature_maps/`：5 个 Hard Test 样本、15 张同层同通道 ECA/LECA 特征图面板及权重表。展示时说“模型响应与通道权重存在可见差异”，不要把热图当作通道语义的直接证明。

## 必须主动说明的审计边界

- 历史 `yolo11.yaml` 已含 8 个隐式 LECA，不能称为 Baseline。
- 历史 train/val 有 5 组精确重复图像，接近 .995 的历史验证指标不作为性能结论。
- 高方差不是反光的同义词，低均值不是螺栓的同义词，深层特征均值不是物理亮度。

完整口径和 3 分钟讲稿见 [`docs/INTERVIEW_EVIDENCE.md`](../docs/INTERVIEW_EVIDENCE.md)。
