# LECA 保守代码审计

审计日期：2026-07-14。本文不改写已发表代码。术语含义固定：**已确认（Confirmed）**由源码、权重或本地审计直接支持；**已观察（Observed）**是实验现象；**假设（Hypothesis）**尚待验证。

## A. 项目结构与实际插入位置

| 项目 | 位置或结论 |
| --- | --- |
| 主源码 | `ultralytics-main/`（Ultralytics 8.3.215） |
| 历史训练入口 | `train_test.py`、`train_abc.py`、`run_ablation_parallel.py` 等 |
| 历史推理/评估 | `detect.py`、`eval_ablation_final.py`、`final_latency_test.py` 等 |
| 历史训练数据 YAML | `ultralytics/cfg/models/datasets/screw.yaml`，指向 `dataset/trainDataV3` |
| Hard Test YAML | `dataset/hardData/YOLODataset/hard_test_set.yaml` |
| 基础模型 YAML | `ultralytics/cfg/models/11/yolo11.yaml` |
| LECA 实现 | `ultralytics/nn/modules/attention.py` 与 `block.py` 都定义同名 `LECA` |

**已确认**：`block.py` 的 `C2f` 从 `attention.py` 导入 LECA，并无条件执行 `self.eca = LECA(c2)`。`C3k2` 继承该结构。因此当前 `yolo11.yaml` 有 8 个隐式注意力模块，索引为 2、4、6、8、13、16、19、22：Backbone 4 个、Neck 3 个、检测头前 1 个。它不是无注意力 Baseline。

**已确认**：历史 Baseline 权重没有 LECA；历史 LECA 权重序列化的是 `attention.LECA`。当前 `block.LECA` 与 `attention.LECA` 是不同 Python 类，部分旧权重只有 alpha/beta/Conv 状态而当前源码声明 gamma。这证明源码与历史权重漂移，不能从当前文件恢复精确论文提交。

## B. 论文公式与当前代码

| 预期计算 | 当前实现 | 结论 |
| --- | --- | --- |
| `mu=mean(X,H,W)`，形状 `[B,C,1,1]` | `x.mean(dim=(2,3), keepdim=True)` | 已确认 |
| `m=mean(X,C)`，形状 `[B,1,H,W]` | `torch.mean(x, dim=1, keepdim=True)` | 已确认 |
| ECA | GAP → `[B,1,C]` Conv1D → sigmoid | 已确认 |
| `var=mean(X²)-mu²` | `mean2-mean*mean` | 已确认 |
| `noise=softplus(var)` | `F.softplus(var)` | 已确认 |
| `w_sup=1/(1+beta*noise)` | 相同 | 已确认 |
| `w_rec=1+alpha*sigmoid(-mu)` | 相同 | 已确认 |
| `corr=mean(sigmoid(-m),H,W)` | 相同，输出 `[B,1,1,1]` | 已确认 |
| `w_bri=1+gamma*corr` | 相同，按通道广播 | 已确认 |
| `out=X*w_eca*w_sup*w_rec*w_bri` | 相同 | 已确认 |

**已确认**：alpha=.02、beta=.04、gamma=.01 是当前每个 LECA 实例独立的 `nn.Parameter`，属于 `model.parameters()`，合成前向/反向检查显示梯度非零。当前 forward 没有 `detach()`、`.item()` 或 NumPy 转换。方差采用总体矩公式，不是无偏方差；ECA 核大小自适应，C=64 为 k=3，C≥128 为 k=5，Conv1D 参数量恰为 k。

**数值风险（已确认的数学性质）**：`mean(X²)-mean(X)²` 未做 `clamp_min(0)`，浮点抵消在 AMP/FP16 下可能产生微小负数；beta 无约束，若变为负值，分母可能靠近零、变负或溢出。当前没有 epsilon、范围约束或 NaN 保护。初始且 beta≥0 时，`w_sup` 正且不大于约 .973，`w_rec∈[1,1.02]`，`w_bri∈[1,1.01]`；训练后上述范围不再保证。

LECA 输入位于 C2f/C3k2 内部卷积、BN/SiLU 之后，不是原始像素，且可包含负值。`corr` 是深层特征全局激活统计，不是物理亮度。

## C. 结构公平性

| 历史 YAML | 参数量 | 显式模块 | 隐式 LECA | 结论 |
| --- | ---: | --- | ---: | --- |
| `yolo11.yaml` | 2,624,140 | 无 | 8 | 不是 Baseline |
| `yolo11EMA.yaml` | 2,316,682 | EMA 2 个 | 8 | 拓扑不匹配 |
| `yolo11SE.yaml` | 2,319,018 | SE 2 个 | 8 | 拓扑不匹配 |
| `yolo11NAM.yaml` | 2,317,226 | NAM 2 个 | 8 | 拓扑不匹配 |
| `yolo11CBAM.yaml` | 2,337,326 | CBAM 2 个 | 8 | 拓扑不匹配 |

**已确认**：对比 YAML 改动 C3k2 重复数、通道声明与模块索引，因此“加注意力后参数/FLOPs 反而下降”是结构被替换所致，不能解释为注意力更高效。代码未发现一致的 Baseline/ECA/LECA FLOPs 统计；`final_latency_test.py` 只测一个 ULECA 权重、GPU 5、模型本体推理，不包含预处理/NMS/传输，不能做跨方法 FPS 比较。

**受控补救（已确认）**：`exp/mechanism-audit` 用同拓扑、同 8 位置构建 Identity Baseline/ECA/LECA，参数量为 2,624,080/2,624,116/2,624,140；理论与实际增量分别为 ECA +36、LECA 相对 Baseline +60、LECA 相对 ECA +24。

## D. 数据审计

历史 `trainDataV3` 有 122 张训练图、37 张验证图、1 类、159 个实例；未发现空标签、坏图、缺标、非法类别或越界框。

**已确认的数据泄漏**：SHA256 发现 8 个完全重复组，其中 5 组跨 train/val：`screw_139↔146`、`141↔134`、`144↔137`、`147↔140`、`189↔185`（完整文件名和哈希在本地 `artifacts/data_audit/`）。另有 2 对训练集内 dHash 距离≤4 的近重复。这阻止将历史验证集指标当作无偏性能证据；数据未被移动或删除。

Hard Test 的 YAML 为兼容接口而把 train/val/test 都指向 `test/`，因此三个键并非独立划分。审计得到 146 张唯一测试图、155 个单类实例。受控 Hard Test 前的 SHA256 检查：`trainDataV3` 159 张图像与 Hard Test 目录的 292 个图像文件项精确交集为 0；这不能排除同场景相邻帧或既往人工调参接触，仍须做近重复与时间顺序审计。

## E. 训练配置审计

历史 `args.yaml` 常见记录为 epochs=200、patience=20、imgsz=640、batch=16、optimizer=auto、预训练 `yolo11n.pt`、AMP=true、deterministic=true、seed=0、cache=false、workers=0 或 8、mosaic=1.0、close_mosaic=10、translate=.1、scale=.5、fliplr=.5，且无 mixup/copy-paste。设备在 0/3/5/7 间变化，未找到论文要求的 seed=42。

风险包括：`optimizer=auto` 随 Ultralytics 版本可能变化；数据 YAML 含绝对路径和旧 Windows 路径；未保留不可变依赖锁、模型 YAML 文本和论文时源码提交；Hard Test 是否参与过历史选权重无法仅凭当前日志证明。

## F. 审计结论

### 已确认的实现事实

1. 当前 LECA 公式主体与目标式一致，形状、广播、参数注册和合成梯度均正常。
2. 当前 `yolo11.yaml` 含 8 个隐式 LECA，比较 YAML 改变了网络结构，历史 Params/FLOPs 不公平。
3. 历史 train/val 有 5 组精确重复，不能使用其高分支撑结论。
4. `paper-original` 已保存原始本地快照；数据、图片、权重、runs、可视化均被 Git 忽略。

### 潜在问题

- beta 无约束、方差未 clamp、AMP 数值稳定性未完全压力验证。
- 两个同名 LECA 类造成注册和权重反序列化歧义。
- 历史论文的确切 YOLO11 YAML/源码提交缺失，无法宣称严格复现。

### 可复现性与公平性风险

- 历史源码与权重漂移、历史配置不完整、设备/版本不一致。
- 历史划分泄漏；Hard Test 的独立性还需近重复与使用时序证据。
- ECA-only YAML 缺失，旧消融脚本的环境变量与当前 LECA forward 不对应。

### 尚未获得支持的机制说法

代码和当前一次 Hard Test 不能证明“高方差就是反光”“低均值就是螺栓”或“深层均值就是物理亮度”。这些最多是待验证统计假设。

### 后续建议

先完成 Hard Test 近重复审计、多种子和重训练消融；每次训练先配置信息检查与 1 epoch smoke。若要修改公式，必须在论文验证完成后另建 `exp/leca-v2`，不污染 `main` 与 `paper-original`。
