# LECA 注意力可视化实现说明

## 1. 目标

本次可视化的目标不是简单展示某一层特征图“哪里激活强”，而是定性说明：

1. baseline 在困难车底图像中会被金属高光、污渍、孔洞等结构干扰，产生误检；
2. 加入 LECA 后，模型在相同置信度阈值下能够更集中地响应真实螺栓区域；
3. LECA 中的三个物理统计分量，即方差抑制、弱特征恢复、亮度校正，以及对应可学习标量，对降低误检和稳定关注区域是有意义的。

最终采用的主图是：

`draw/results/0001_b1214a5d_conf050/compare/00_key_layer16_summary.jpg`

该图在 `conf=0.5` 下对比 baseline 与 LECA7：

- baseline：检测出一个真实螺栓，同时出现一个误检；
- LECA7：只保留真实螺栓，误检被抑制；
- layer16 的 Grad-CAM 显示 LECA7 对真实螺栓区域的响应更集中，适合作为论文或汇报主图。

## 2. 调研依据

### 2.1 Grad-CAM

调研文献：

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.  
  链接：https://arxiv.org/abs/1610.02391

采用原因：

Grad-CAM 使用目标类别或目标分数对中间特征图求梯度，将梯度在空间维度上全局平均后作为通道权重，再对特征图加权求和。它回答的是“为了产生这个检测目标，网络依赖了哪些空间区域”，比普通特征能量图更适合证明误检来源和真实目标关注区域。

本实现中没有直接使用分类类别分数，而是使用 YOLO 原始输出中的目标置信度分数 `pred[0, 4, raw_index]` 作为反传目标。这样热力图对应的是具体检测框，而不是整张图的泛化激活。

### 2.2 Grad-CAM++ 与 Eigen-CAM

调研文献：

- Chattopadhay et al., *Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks*, WACV 2018.  
  链接：https://arxiv.org/abs/1710.11063
- Muhammad and Yeasin, *Eigen-CAM: Class Activation Map using Principal Components*, IJCNN 2020.  
  链接：https://arxiv.org/abs/2008.00299

结论：

最开始尝试了类似 Eigen-CAM/通道聚合的方式，把中间特征压成二维热力图。它不需要反传，速度快，但问题很明显：图中大量金属边缘、车底纹理、低层结构都会变红，不能说明“哪个区域真正支撑了最终检测结果”。因此最终放弃了单纯特征能量图，改成目标驱动 Grad-CAM。

### 2.3 Ultralytics 特征图可视化

调研资料：

- Ultralytics Predict mode 文档，包含 `visualize=True` 的特征图保存入口。  
  链接：https://docs.ultralytics.com/modes/predict/
- 本地源码中也有 `ultralytics/utils/plotting.py::feature_visualization()`，它会保存每层若干通道的 feature map 网格。

结论：

Ultralytics 自带 `visualize=True` 更适合查看通道特征是否正常，不适合作为本文的核心定性证据。原因是它保存的是中间通道图，缺少“该响应是否服务于最终螺栓检测框”的目标约束。本文需要解释误检降低，因此必须把最终检测目标和中间层响应关联起来。

### 2.4 PyTorch Hook 机制

调研资料：

- PyTorch `register_forward_hook` 文档。  
  链接：https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- PyTorch `Tensor.retain_grad` 文档。  
  链接：https://pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html

采用原因：

YOLO 推理流程中，中间层输出不会自动保留梯度。为了计算 Grad-CAM，需要：

1. 对指定层注册 forward hook，保存该层输出；
2. 对该层输出调用 `retain_grad()`；
3. 对指定检测目标分数执行 `backward()`；
4. 读取该层输出的梯度，计算 Grad-CAM。

## 3. 为什么选择 layer16

这次默认可视化层为：

```python
DEFAULT_LAYERS = [13, 16, 19]
```

对样例图实际观察后：

- layer13：偏底层/中层纹理，金属边缘、孔洞、高光仍然较多，适合作为补充，不适合作为主图；
- layer16：既保留螺栓局部结构，又已经接近检测头，Grad-CAM 最清晰；baseline 误检区域仍有明显响应，而 LECA7 更集中在真实螺栓；
- layer19：该样例上梯度响应较弱，热力图信息不足，不建议作为正文主图。

因此建议论文正文主图使用 layer16，其他层作为附录或辅助分析。

## 4. 代码加入位置

新增脚本：

```text
draw/scripts/gradcam_attention_compare.py
```

结果目录：

```text
draw/results/0001_b1214a5d_conf050/
├── baseline/
├── LECA7/
└── compare/
```

其中：

- `baseline/`：baseline 模型的预测图、真实目标 Grad-CAM、误检目标 Grad-CAM、局部 crop；
- `LECA7/`：LECA7 模型的预测图、真实目标 Grad-CAM、局部 crop、可学习标量 CSV；
- `compare/`：汇总对比图和辅助指标。

## 5. 关键实现说明

### 5.1 默认输入、权重和层号

位置：

```text
draw/scripts/gradcam_attention_compare.py:22-27
```

内容：

```python
DEFAULT_IMAGE = ROOT / "dataset/hardData/YOLODataset/images/train/0001_b1214a5d.jpg"
DEFAULT_BASELINE = ROOT / "runs/detect/car_bolt_baseline/weights/best.pt"
DEFAULT_LECA = ROOT / "runs/detect/car_bolt_LECA7/weights/best.pt"
DEFAULT_OUT = ROOT / "draw/results"
DEFAULT_LAYERS = [13, 16, 19]
```

### 5.2 兼容旧 baseline 权重

位置：

```text
draw/scripts/gradcam_attention_compare.py:39-48
```

原因：

当前工程里的 `C2f/C3k2.forward()` 已经加入了 `self.eca(out)`，但 baseline 权重来自改进前结构，部分层没有 `eca` 属性。为了不修改权重和模型源码，脚本运行时临时 patch：

```python
return self.eca(out) if hasattr(self, "eca") else out
```

这样 baseline 没有注意力模块时按原始输出走，LECA7 有注意力模块时正常调用 `eca`。

### 5.3 YOLO 输入预处理与坐标还原

位置：

```text
draw/scripts/gradcam_attention_compare.py:51-101
```

实现内容：

- `letterbox()`：按 YOLO 推理方式 resize + padding 到 640；
- `to_tensor()`：BGR 转 RGB，归一化到 `[0,1]`，并开启 `requires_grad`；
- `original_to_letterbox_box()` 和 `letterbox_to_original_box()`：在原图坐标与 letterbox 坐标之间转换。

这样可以保证 Grad-CAM 热力图最终准确贴回原始 1920x1080 图像。

### 5.4 目标选择：真实螺栓与误检框

位置：

```text
draw/scripts/gradcam_attention_compare.py:197-227
```

逻辑：

1. 从 YOLO 原始输出中取 `scores = pred[0, 4, :]`；
2. 将预测框转换为 `xyxy`；
3. 计算所有预测框与 GT 螺栓框的 IoU；
4. 选择 `IoU > 0.2` 且 `score >= conf` 的预测作为 true bolt；
5. 选择 `IoU < 0.05` 且 `score >= conf` 的高分预测作为 false positive。

本次使用：

```bash
--conf 0.5
```

样例结果中，baseline 有一个 TP 和一个 FP，LECA7 只有一个 TP，因此这张图非常适合说明误检抑制。

### 5.5 注册中间层 Hook

位置：

```text
draw/scripts/gradcam_attention_compare.py:182-194
```

核心代码：

```python
def forward_hook(_module, _inp, out, idx=layer_idx):
    activations[idx] = out
    out.retain_grad()

handles.append(layer.register_forward_hook(forward_hook))
```

作用：

- 捕获 layer13、layer16、layer19 的输出特征；
- 保留这些中间特征的梯度；
- 后续通过 `score.backward()` 得到每一层对目标检测分数的贡献。

### 5.6 Grad-CAM 计算

位置：

```text
draw/scripts/gradcam_attention_compare.py:230-266
```

核心代码：

```python
score = pred[0, 4, target.raw_index]
score.backward()

grad = act.grad
weights = grad.mean(dim=(2, 3), keepdim=True)
cam = torch.relu((weights * act).sum(dim=1, keepdim=True))[0, 0]
```

含义：

- `score` 是某一个检测框的置信度分数；
- `grad` 表示该检测分数对中间层特征的梯度；
- `weights` 是每个通道的重要性；
- `weights * act` 得到通道加权后的空间响应；
- `relu` 只保留对该目标有正贡献的区域。

为了避免多个目标共用同一计算图导致显存和运行时间问题，脚本对每个目标重新前向一次，再单独反传。

### 5.7 热力图叠加与阈值过滤

位置：

```text
draw/scripts/gradcam_attention_compare.py:124-146
```

采用稀疏叠加：

```python
mask = (heat >= threshold).astype(np.float32)[..., None]
```

本次默认阈值设置为：

```python
--overlay-threshold 0.62
```

原因：

上一版低阈值会让车底纹理、金属边缘和低置信响应大量覆盖原图，看起来“到处都红”。提高显示阈值后，只保留高响应区域，更适合展示注意力集中效果。

### 5.8 输出组织

位置：

```text
draw/scripts/gradcam_attention_compare.py:335-407
```

脚本会自动创建：

```text
draw/results/{image_stem}_conf050/
├── baseline/
│   ├── predictions_conf050.jpg
│   ├── gradcam_true_bolt_layer16.jpg
│   ├── gradcam_false_positive_layer16.jpg
│   └── crop_*.jpg
├── LECA7/
│   ├── predictions_conf050.jpg
│   ├── gradcam_true_bolt_layer16.jpg
│   ├── crop_*.jpg
│   └── learned_LECA_scalars.csv
└── compare/
    ├── 00_key_layer16_summary.jpg
    ├── 01_predictions_baseline_vs_LECA7_conf050.jpg
    ├── 02_gradcam_true_bolt_layers_baseline_vs_LECA7.jpg
    ├── 03_baseline_false_positive_gradcam_layers.jpg
    └── gradcam_attention_metrics.csv
```

其中 `00_key_layer16_summary.jpg` 是最适合正文展示的图。

## 6. 运行方式

默认运行：

```bash
cd /home/lzx/car_bolt_detection/ultralytics-main
/home/lzx/miniconda3/envs/yolo/bin/python draw/scripts/gradcam_attention_compare.py
```

指定图片：

```bash
/home/lzx/miniconda3/envs/yolo/bin/python draw/scripts/gradcam_attention_compare.py \
  --image dataset/hardData/YOLODataset/images/train/0001_b1214a5d.jpg \
  --conf 0.5
```

只看 layer16：

```bash
/home/lzx/miniconda3/envs/yolo/bin/python draw/scripts/gradcam_attention_compare.py \
  --layers 16 \
  --conf 0.5
```

调节显示阈值：

```bash
/home/lzx/miniconda3/envs/yolo/bin/python draw/scripts/gradcam_attention_compare.py \
  --overlay-threshold 0.62
```

阈值越高，显示越稀疏；阈值越低，显示越完整但更容易显得噪声大。

## 7. 可用于论文的表述建议

可以写成：

> 为定性分析改进注意力机制对复杂车底场景的作用，本文采用目标驱动 Grad-CAM 对改进前后的 YOLO 模型进行可视化。不同于直接显示中间特征图能量，本文以最终检测框置信度作为反传目标，提取 neck/head 前的中间层响应，从而观察模型对具体螺栓检测结果的空间依据。在困难样本中，baseline 在 `conf=0.5` 下同时激活真实螺栓与金属污渍区域并产生误检；加入 LECA 后，误检框被抑制，layer16 的高响应区域更加集中于真实螺栓附近。该现象说明方差抑制、弱特征恢复和亮度校正分量能够改善模型在金属高光、低照度和弱纹理场景下的注意力分配。

需要注意：

- 不建议说 Grad-CAM 是严格的因果证明；
- 更稳妥的说法是“定性说明”“可视化证据”“模型响应更集中”；
- 最终证明仍应结合检测指标、误检案例统计和消融实验。
