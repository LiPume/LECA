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

六个缺失组合均先通过 1 epoch smoke，再独立完成最长 200 epoch、patience=20 的训练；ECA 与 Full 使用此前同协议 seed=42 权重。八组仅在训练完成后统一评估 Hard Test。146 张场景初标只查看原图联系表，不读取逐图模型预测；图片联系表和逐图统计均只保存在 ignored artifacts。
