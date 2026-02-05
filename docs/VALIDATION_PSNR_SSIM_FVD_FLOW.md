# 训练时验证 PSNR / SSIM / FVD 全流程与代码对应

## 一、触发时机

验证在两种时机执行：

| 时机 | 条件 | 代码位置 |
|------|------|----------|
| **Step 0** | 训练开始、加载 checkpoint 之前，保证 step 0 = 预训练模型指标 | `diffsynth/diffusion/runner.py` 第 102–104 行 |
| **周期性** | 每 `save_steps` 一次（如 50、100） | `diffsynth/diffusion/runner.py` 第 164–170 行 |

```python
# runner.py L102-L104: Step 0 验证（预训练模型）
if val_dataset is not None and getattr(args, "val_num_videos", 0) > 0:
    evaluate_on_validation(accelerator, model, val_dataset, args, global_step=0)

# runner.py L164-L170: 周期性验证
if (val_dataset is not None and save_steps is not None
    and model_logger.num_steps % save_steps == 0):
    evaluate_on_validation(accelerator, model, val_dataset, args, global_step=model_logger.num_steps)
```

---

## 二、验证集构建（固定样本，与 shuffle 无关）

验证集用 **稳定 id（clip_id / 路径）** 选样本，保证不同 step、resume 都用同一批样本。

| 步骤 | 说明 | 代码位置 |
|------|------|----------|
| 1 | 若存在 `{output_path}/val/val_ids.json`，从中读取 `val_ids` | `train.py` 第 263–273 行 |
| 2 | 否则按数据集顺序取前 N 个**不重复**的 `get_stable_id(i)` 作为 `val_ids` | `train.py` 第 274–283 行 |
| 3 | 用 `dataset.get_indices_for_ids(val_ids)` 得到下标，构建 `Subset(dataset, val_indices)` | `train.py` 第 284–289 行 |
| 4 | 将本次使用的 `val_ids` 写入 `val_ids.json`，供下次/resume 复用 | `train.py` 第 285–288 行 |

```python
# train.py L258-L289: 验证集构建
val_dataset = None
if getattr(args, "val_num_videos", 0) and args.val_num_videos > 0:
    from torch.utils.data import Subset
    num_val = min(args.val_num_videos, len(dataset))
    # 1) 尝试从 val_ids.json 加载
    if val_ids_path and os.path.isfile(val_ids_path):
        saved = json.load(...)
        val_ids = [tuple(x) if isinstance(x, list) else x for x in saved.get("val_ids", [])]
    # 2) 否则按顺序收集前 N 个不重复 id
    if not val_ids:
        for i in range(len(dataset)):
            sid = dataset.get_stable_id(i)
            if sid not in seen:
                val_ids.append(sid)
            if len(val_ids) >= num_val: break
    # 3) id -> 下标，建 Subset
    val_indices = dataset.get_indices_for_ids(val_ids)[:num_val]
    val_dataset = Subset(dataset, val_indices)
```

- 稳定 id 定义：`diffsynth/core/data/unified_dataset.py` 中 `get_stable_id(i)`、`get_indices_for_ids(ids)`。

---

## 三、单次验证流程：`evaluate_on_validation`

入口：`diffsynth/diffusion/runner.py` 第 178 行 `def evaluate_on_validation(...)`。

### 3.1 仅主进程跑、取 pipe

- 非 main process 直接 return。
- 从 model 取 `pipe`，没有则跳过。

```python
# runner.py L191-L204
if not accelerator.is_main_process:
    return
# ...
pipe = getattr(unwrapped_model, "pipe", None)
if pipe is None:
    return
```

### 3.2 记录并保存验证样本 id

- 从 `val_dataset`（Subset）得到 `val_indices`，若有 `get_stable_id` 则得到 `val_ids`。
- 打印并写入 `{output_path}/val/val_ids.json`。

```python
# runner.py L206-L226
if isinstance(val_dataset, torch.utils.data.Subset):
    base_dataset = val_dataset.dataset
    val_indices = list(val_dataset.indices)
    val_ids = [base_dataset.get_stable_id(i) for i in val_indices] if hasattr(...) else None
# ...
with open(ids_path, "w") as f:
    json.dump(to_save, f, indent=2)
```

### 3.3 验证 DataLoader

- 使用 `val_dataset`，`shuffle=False`，保证顺序固定。

```python
# runner.py L238-L244
dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=lambda x: x[0],
)
```

### 3.4 逐样本：构造输入 → 推理 → 对齐帧数 → 存视频 + 转张量

对每个 batch（一条样本）：

| 步骤 | 说明 | 代码位置 |
|------|------|----------|
| 1 | 从 data 取 GT 视频 `gt_video = data["video"]` | `runner.py` L266 |
| 2 | `get_pipeline_inputs(data)` 得到 `inputs_shared/inputs_posi` | `runner.py` L269 |
| 3 | 固定 `num_frames`（4n+1）、`target_len = num_frames - 4`，与 inference 一致 | `runner.py` L271–293 |
| 4 | 将 animate_* 等截断/填充到 `target_len`，拼出 `pipe_kwargs` | `runner.py` L290–316 |
| 5 | `generated_video = pipe(**pipe_kwargs)` 生成预测视频 | `runner.py` L317 |
| 6 | 按 `min(len(gt), len(pred))` 对齐帧数，得到 `gt_video_aligned`、`pred_video_aligned` | `runner.py` L319–324 |
| 7 | 可选：保存 `sample-XXXX_gt.mp4`、`sample-XXXX_pred.mp4` 到 `{output_path}/val/step-{step}/` | `runner.py` L327–336 |
| 8 | `video_to_tensor` 转成 (C,T,H,W)，追加到 `real_videos`、`fake_videos` | `runner.py` L338–341 |

```python
# runner.py L266-L341 核心循环
for idx, data in enumerate(dataloader):
    gt_video = data["video"]
    inputs_shared, inputs_posi, inputs_nega = unwrapped_model.get_pipeline_inputs(data)
    # 固定 num_frames、target_len，截断/填充 animate_*
    val_num_frames = ...  # args.num_frames，并保证 4n+1
    target_len = max(1, val_num_frames - 4)
    # ...
    generated_video = pipe(**pipe_kwargs)
    aligned_frames = min(len(gt_video), len(generated_video))
    gt_video_aligned = gt_video[:aligned_frames]
    pred_video_aligned = generated_video[:aligned_frames]
    save_video(gt_video_aligned, gt_path, ...)
    save_video(pred_video_aligned, pred_path, ...)
    gt_tensor = video_to_tensor(gt_video_aligned)
    pred_tensor = video_to_tensor(pred_video_aligned)
    real_videos.append(gt_tensor)
    fake_videos.append(pred_tensor)
```

- `video_to_tensor`：`diffsynth/metrics/video_metrics.py` 第 26–42 行，输出 (C, T, H, W)，数值 [0, 1]。

---

## 四、指标计算：PSNR / SSIM / FVD

在收集完本步所有验证样本的 `real_videos`、`fake_videos` 后，调用统一入口计算指标。

### 4.1 调用入口

```python
# runner.py L349-L364
metrics_dict = compute_video_metrics(
    real_videos,
    fake_videos,
    device=device,
    data_range=1.0,
    compute_fvd=True,
    fvd_batch_size=max(1, getattr(args, "val_batch_size", 1)),
    max_fvd_samples=getattr(args, "val_num_videos", None),
)
metrics = {
    "val/psnr": metrics_dict["psnr"],
    "val/ssim": metrics_dict["ssim"],
    "val/fvd": metrics_dict["fvd"],
}
accelerator.log(metrics, step=global_step)
```

- 实现：`diffsynth/metrics/video_metrics.py` 中 `compute_video_metrics`（第 70–119 行）。

### 4.2 PSNR

- 公式：`10 * log10(data_range^2 / MSE)`，MSE 在整段视频上算。
- 代码：`video_metrics.py` 第 45–54 行 `compute_psnr(pred, target, data_range)`；在 `compute_video_metrics` 里对每对 (real, fake) 算 PSNR 再取平均（L95–98、L103）。

### 4.3 SSIM

- 逐帧 SSIM 再对时间取平均，依赖 `torchmetrics.functional.image.structural_similarity_index_measure`。
- 代码：`video_metrics.py` 第 57–67 行 `compute_ssim_video`；在 `compute_video_metrics` 里对每对视频算 SSIM 再取平均（L99–100、L104）。

### 4.4 FVD

- 使用 I3D 特征，在 `[0,1]` 转成 `[-1,1]` 后送入 FVD 实现；可选限制参与 FVD 的样本数 `max_fvd_samples`。
- 代码：`video_metrics.py` 第 108–118 行，调用 `diffsynth.metrics.fvd_metric.compute_fvd`（若存在）；张量维度与 batch 处理在同一文件内完成。

---

## 五、结果记录与 best 更新

| 内容 | 位置 | 说明 |
|------|------|------|
| WandB | `runner.py` L364 | `accelerator.log(metrics, step=global_step)` 打 val/psnr、val/ssim、val/fvd |
| 控制台 | `runner.py` L366 | 打印当前 step 的 PSNR/SSIM/FVD |
| Best 记录 | `runner.py` L369–402 | 读/写 `{output_path}/val/best_metrics.json`，更新并打印 best_psnr、best_ssim、best_fvd 及其 step |

---

## 六、相关参数（wan_parser / example_train.sh）

| 参数 | 含义 | 默认/示例 |
|------|------|-----------|
| `--val_num_videos` | 验证视频条数 | 8、16 |
| `--val_batch_size` | 验证时 batch（含 FVD 的 batch 划分） | 1 |
| `--val_inference_steps` | 验证推理步数 | 20 |
| `--val_cfg_scale` | 验证 CFG | 1.0 |
| `--val_seed` | 验证随机种子 | 0 |
| `--num_frames` | 验证用的帧数（4n+1），与 inference 对齐 | 49、81 |
| `--save_steps` | 多少 step 做一次验证并存 checkpoint | 50 |

---

## 七、流程简图

```
train.py 启动
  → 构建 dataset、val_dataset（val_ids / val_indices）
  → launch_training_task()

runner.py launch_training_task()
  → [Step 0] evaluate_on_validation(..., global_step=0)
  → [可选] 加载 resume_checkpoint
  → 训练循环
       → 每 save_steps: evaluate_on_validation(..., global_step=num_steps)

evaluate_on_validation()
  → 仅 main process，取 pipe
  → 写 val_ids.json
  → DataLoader(val_dataset, shuffle=False)
  → for each sample:
        get_pipeline_inputs → 固定 num_frames/target_len → pipe(**kwargs) → 对齐帧数
        → save_video(gt, pred) 到 val/step-{step}/
        → video_to_tensor → real_videos, fake_videos
  → compute_video_metrics(real_videos, fake_videos) → PSNR, SSIM, FVD
  → accelerator.log(val/psnr, val/ssim, val/fvd)
  → 更新并保存 best_metrics.json，打印 best
```

---

## 八、inference 时算指标（可选）

`inference.py` 中若传 `--compute_metrics`，会在推理结束后对收集的 (gt, pred) 对调用同一套 `compute_video_metrics`，并写 `output_dir/metrics.json`；多 GPU 时通过 `gather_video_pairs_to_rank0` 在 rank 0 上算。指标定义与训练验证一致，见 `diffsynth/metrics/video_metrics.py`。
