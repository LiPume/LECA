# 命令记录

除非另有说明，以下命令均在 `/home/lzx/car_bolt_detection` 执行。数据集、图片、权重、训练输出和可视化均在本地且被 Git 忽略。

## 2026-07-14：环境与安全审计

```bash
pwd
conda run -n yolo python -V
conda run -n yolo python -c "import torch; ..."
nvidia-smi || true
find . -type f -size +50M -not -path './.git/*'
grep -RInE '(BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|github_pat_|ghp_|AKIA|api[_-]?key|secret[_-]?key|password[ ]*=)' . ... || true
conda env export -n yolo --no-builds | sed '/^prefix:/d' > environment.yml
```

本地数据与模型审计：

```bash
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/audit_dataset.py \
  ultralytics-main/ultralytics/cfg/models/datasets/screw.yaml \
  --output artifacts/data_audit/trainDataV3_summary.json
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/audit_dataset.py \
  ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml \
  --output artifacts/data_audit/hard_test_summary.json
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/audit_model_build.py
```

上述审计发现历史 train/val 有 5 组精确重复，且当前 `yolo11.yaml` 含 8 个隐式 LECA。输出只落地到本地 `artifacts/data_audit/`。

## 2026-07-14：Git 安全快照

```bash
git init
git branch -M main
git remote add origin git@github.com:LiPume/LECA.git
git add .
git commit -m "first commit"
git tag -a paper-original -m "Published LECA implementation before audit"
git push -u origin main
git push origin paper-original
```

远端为空，非强推送成功。暂存前已验证媒体、数据集、权重、训练输出、artifacts 和可视化均被忽略。

## 2026-07-14：受控机制审计与 Smoke Test

```bash
git switch -c exp/mechanism-audit
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/train_mechanism_smoke.py \
  --model baseline --device 0 --batch 16 --workers 4
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/train_mechanism_smoke.py \
  --model eca --device 0 --batch 16 --workers 4
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/train_mechanism_smoke.py \
  --model leca --name leca_stats_retry --device 0 --batch 16 --workers 4
```

该分支以显式 Identity/ECA/LECA C3k2 包装器建立拓扑匹配模型，不是论文复现。第一次 LECA 统计因 hook 注册早于 Ultralytics 复制训练模型而未记录特征聚合；已修复为 `on_pretrain_routine_end` 后注册，并保留原输出、用 `leca_stats_retry` 单独重跑，未覆盖任何结果。

```bash
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/evaluate_branch_sensitivity.py \
  --checkpoint runs_repro/mechanism_smoke/leca_stats_retry/weights/best.pt --device 0
```

1 epoch 后 Full/Var-off/Rec-off/Bri-off 均无检测，记录为未收敛限制，不能解释为分支无价值。

## 2026-07-14：Hard Test 受控评估与特征图

```bash
conda run -n yolo env PYTHONPATH=ultralytics-main python -m py_compile \
  tools/evaluate_hard_test.py tools/visualize_eca_leca_features.py
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/evaluate_hard_test.py
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/visualize_eca_leca_features.py
```

输出：Hard Test 汇总在本地 `runs_repro/hard_test_controlled/hard_test_summary.csv`；特征图面板与索引在本地 `artifacts/visualizations/hard_feature_maps/`。二者均被 Git 忽略。

Hard Test 精确重复检查：

```bash
python - <<'PY'
# 对 trainDataV3 与 hardData/YOLODataset 中所有图像计算 SHA256 并求交集
PY
```

结果：`trainDataV3_images=159`、`hard_images=292`、`exact_sha256_overlap=0`。该检查只证明无字节级重复，不代替近重复或使用时序审计。

## 2026-07-14：统计语义、八组合消融与场景初标

```bash
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/run_semantic_diagnostics.py --device 3 --brightness-images 24
conda run -n yolo python tools/build_hard_case_annotation_sheets.py
conda run -n yolo python tools/write_initial_hard_case_labels.py
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/train_factorial_ablation.py \
  --combination <var|rec|bri|var_rec|var_bri|rec_bri> --mode smoke --device <0|2|3>
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/train_factorial_ablation.py \
  --combination <var|rec|bri|var_rec|var_bri|rec_bri> --mode full --device <0|2|3>
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/evaluate_factorial_ablation.py --device 3 --seed 42
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/evaluate_hard_case_groups.py --device 3 --seed 42
```

六个缺失组合均先通过 1 epoch smoke，再独立完成最长 200 epoch、patience=20 的训练；ECA 与 Full 使用此前同协议 seed=42 权重。八组仅在训练完成后统一评估 Hard Test。146 张场景初标使用不显示预测框的原图联系表，但标注者此前参与过少量案例审计，因此不是严格盲法；图片联系表和逐图统计均只保存在 ignored artifacts。

## 2026-07-14：LECA 插入位置消融

首次构建因脚本加载 Conda 中官方 Ultralytics、未加载项目自定义模块而报 `KeyError: C3k2Baseline/C3k2LECA`。错误未绕过；随后仅在实验脚本中固定项目内 `ultralytics-main` 导入路径，未修改 LECA 核心实现。

```bash
conda run -n yolo python tools/train_placement_ablation.py --placement <none0|backbone4|fusion4|scales3|full8> --mode build --device 0
conda run -n yolo python tools/train_placement_ablation.py --placement <backbone4|fusion4|scales3> --mode smoke --device 0
conda run -n yolo python tools/train_placement_ablation.py --placement <backbone4|fusion4|scales3> --mode full --device 0
conda run -n yolo python tools/evaluate_placement_ablation.py --device 0 --seed 42 --batch 16
```

三组中间配置均通过 1 epoch smoke 后独立训练，Hard Test 仅在全部训练完成后统一评估。Full 8 处四项指标最高；小型汇总写入 `reports/placement_ablation_results.csv`，权重、训练图和评估图保存在 ignored `runs_repro/`。

## 2026-07-15：逐层实际因子与分支中性化

```bash
conda run -n yolo python tools/summarize_leca_layer_factors.py --device 0
conda run -n yolo python tools/plot_leca_layer_factors.py
conda run -n yolo python tools/evaluate_hard_branch_neutralization.py --device 0
```

使用 Full LECA seed=42 最优权重遍历全部 146 张 Hard Test，只保存逐层聚合统计，不保存原始逐通道特征。首次汇总得到 147 次 `corr/w_bri`，定位为 Ultralytics 首次前向的 dummy warm-up 被 hook 捕获；该异常未静默忽略。脚本改为先完成模型 warm-up、再注册 hooks，重新生成后逐层计数均严格为 146。

输出为 `reports/leca_layer_factor_summary_seed42.csv` 和 `reports/hard_branch_neutralization_seed42.csv`。逐层曲线保存在 ignored 的 `artifacts/visualizations/leca_layer_factors/`；中性化评估输出保存在 ignored 的 `runs_repro/mechanism_diagnostics/`。中性化只在内存中将 beta/alpha/gamma 置零，未覆盖权重文件，也不替代独立重训练消融。

## 2026-07-15：答辩稿 YOLO11 结构核验

```bash
# 以 1×3×640×640 零张量只读前向，对关键层注册临时 shape hooks
conda run --no-capture-output -n yolo python - <<'PY'
# 构建 configs/yolo11_audit_leca.yaml，打印 2/4/6/8/10/13/16/19/22/23 层输出
PY

# 读取已训练 checkpoint 的 Detect 配置
conda run --no-capture-output -n yolo python - <<'PY'
# 打印 nc、reg_max、stride 和三尺度输入通道
PY
```

确认最终 Detect 输入为第 16/19/22 层，形状分别为 `64×80×80`、`128×40×40`、`256×20×20`，合计 8400 个预测位置；已训练权重为单类别、`reg_max=16`、stride=8/16/32。该检查没有训练、保存权重或修改模型状态。

## 2026-07-15：P3/P4/P5 代表样本可视化

```bash
conda run --no-capture-output -n yolo python tools/visualize_pyramid_features.py --device 0
```

对事先已选为检出增益、误检减少和仍有局限的 `0098/0061/0010` 三个案例，捕获已训练 Full LECA 第 16/19/22 层的实际输出。所有层统一使用全通道 RMS 聚合，不按视觉效果挑选通道；使用 `rect=False` 固定 640 square-letterbox，去除 padding 后映射回原图。图片保存至 ignored 的 `artifacts/visualizations/pyramid_features/`，仅提交生成脚本和不含图像内容的汇总 CSV。

首次尝试分类 Grad-CAM 时，PyTorch 报错 `Invalid device string: '0'`。该错误未绕过；只在新可视化脚本中将数字设备参数显式解析为 `cuda:0`，随后重新运行：

```bash
conda run --no-capture-output -n yolo python tools/visualize_pyramid_gradcam.py --device 0
```

新版图片保存至 ignored 的 `artifacts/visualizations/pyramid_explainable/`：上排使用双线性/三次插值显示全通道 RMS 响应，下排从每个尺度最高原始分类分数反向传播 Grad-CAM。低于 `.01` 的尺度明确标为未形成有效候选，不对微小梯度噪声强行归一化。另输出去除 letterbox padding 的原始通道值面板，通道按有效区域空间标准差固定选 Top-6，而非人工挑图。
