"""
Generic video quality metrics (PSNR, SSIM, LPIPS, MUSIQ, FVD) using pyiqa for use in both 
training validation and inference. Supports multi-GPU: gather (gt, pred) pairs from 
all ranks to rank 0 and compute metrics on rank 0.
"""
import math
import os
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch
from PIL import Image

try:
    import pyiqa
    _pyiqa_available = True
    _pyiqa_metric_cache = {}  # 按 (metric_name, device) 缓存模型
except ImportError:
    _pyiqa_available = False
    _pyiqa_metric_cache = {}
except Exception:
    _pyiqa_available = False
    _pyiqa_metric_cache = {}

try:
    # 尝试导入 FVD 模块
    from diffsynth.metrics.fvd_metric import compute_fvd as _compute_fvd  # type: ignore[import]
    import sys
    print(f"[INFO] Successfully imported compute_fvd from diffsynth.metrics.fvd_metric", file=sys.stderr)
except ImportError as e:
    import sys
    print(f"[WARN] Failed to import compute_fvd from diffsynth.metrics.fvd_metric: {e}", file=sys.stderr)
    print(f"[WARN] ImportError details: {type(e).__name__}: {e}", file=sys.stderr)
    _compute_fvd = None
except Exception as e:
    import sys
    print(f"[WARN] Unexpected error importing compute_fvd: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    _compute_fvd = None


def video_to_tensor(video: List[Union[Image.Image, np.ndarray]]) -> torch.Tensor:
    """
    Convert a list of frames (PIL.Image or HWC numpy) to tensor (C, T, H, W), value range [0, 1].
    """
    frames = []
    for image in video:
        if isinstance(image, Image.Image):
            arr = torch.as_tensor(np.array(image, dtype=np.float32))
        else:
            arr = torch.as_tensor(np.asarray(image, dtype=np.float32))
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.permute(2, 0, 1)
        arr = arr / 255.0
        frames.append(arr)
    return torch.stack(frames, dim=1)


def _get_pyiqa_metric(metric_name: str, device: Union[torch.device, str]):
    """获取或创建 pyiqa 指标模型（带缓存）"""
    if not _pyiqa_available:
        return None
    
    device = torch.device(device) if isinstance(device, str) else device
    cache_key = (metric_name, str(device))
    
    global _pyiqa_metric_cache
    if cache_key not in _pyiqa_metric_cache:
        try:
            _pyiqa_metric_cache[cache_key] = pyiqa.create_metric(metric_name, device=device)
        except Exception as e:
            import sys
            print(f"[WARN] Failed to create pyiqa metric '{metric_name}': {e}", file=sys.stderr)
            return None
    return _pyiqa_metric_cache[cache_key]


def compute_metric_video(
    pred: torch.Tensor,
    target: torch.Tensor,
    metric_name: str,
    device: Union[torch.device, str],
) -> float:
    """
    使用 pyiqa 计算视频指标（逐帧计算然后平均）。
    
    Args:
        pred: (C, T, H, W) tensor in [0, 1] range
        target: (C, T, H, W) tensor in [0, 1] range
        metric_name: pyiqa 支持的指标名称，如 'psnr', 'ssim', 'lpips', 'ms_ssim' 等
        device: 计算设备
    
    Returns:
        平均指标值
    """
    if not _pyiqa_available:
        return float("nan")
    
    metric = _get_pyiqa_metric(metric_name, device)
    if metric is None:
        return float("nan")
    
    device = torch.device(device) if isinstance(device, str) else device
    
    # 对齐帧数
    num_frames = min(pred.shape[1], target.shape[1])
    pred = pred[:, :num_frames].to(device)
    target = target[:, :num_frames].to(device)
    
    # pyiqa 期望输入格式为 (N, C, H, W)，需要逐帧处理
    metric_vals = []
    with torch.no_grad():
        for t in range(num_frames):
            pred_frame = pred[:, t]  # (C, H, W)
            target_frame = target[:, t]  # (C, H, W)
            
            # 添加 batch 维度: (C, H, W) -> (1, C, H, W)
            pred_batch = pred_frame.unsqueeze(0)
            target_batch = target_frame.unsqueeze(0)
            
            try:
                score = metric(pred_batch, target_batch)
                if isinstance(score, torch.Tensor):
                    score_val = score.item()
                else:
                    score_val = float(score)
                metric_vals.append(score_val)
            except Exception:
                continue
    
    if not metric_vals:
        return float("nan")
    return float(np.mean(metric_vals))


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: Union[torch.device, str],
    data_range: float = 1.0,
) -> float:
    """PSNR between two videos (C, T, H, W) using pyiqa."""
    return compute_metric_video(pred, target, 'psnr', device)


def compute_ssim_video(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: Union[torch.device, str],
    data_range: float = 1.0,
) -> float:
    """SSIM over video (C, T, H, W) via per-frame SSIM then mean using pyiqa."""
    return compute_metric_video(pred, target, 'ssim', device)


def compute_lpips_video(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: Union[torch.device, str],
    net: str = 'alex',
) -> float:
    """
    LPIPS over video (C, T, H, W) via per-frame LPIPS then mean using pyiqa.
    Input tensors should be in [0, 1] range.
    """
    # pyiqa 的 LPIPS 默认使用 alex，可以通过 metric_kwargs 配置
    metric_name = 'lpips'
    return compute_metric_video(pred, target, metric_name, device)


def compute_musiq_video(
    pred: torch.Tensor,
    device: Union[torch.device, str],
) -> float:
    """
    MUSIQ (no-reference) over video (C, T, H, W) via per-frame MUSIQ then mean using pyiqa.
    MUSIQ is a no-reference metric, so it only needs the predicted video.
    Input tensor should be in [0, 1] range.
    """
    if not _pyiqa_available:
        return float("nan")
    
    metric = _get_pyiqa_metric('musiq', device)
    if metric is None:
        return float("nan")
    
    device = torch.device(device) if isinstance(device, str) else device
    
    # MUSIQ 是 no-reference，只需要 pred
    pred = pred.to(device)
    num_frames = pred.shape[1]
    
    # pyiqa 的 no-reference 指标期望输入格式为 (N, C, H, W)，需要逐帧处理
    musiq_vals = []
    with torch.no_grad():
        for t in range(num_frames):
            pred_frame = pred[:, t]  # (C, H, W)
            
            # 添加 batch 维度: (C, H, W) -> (1, C, H, W)
            pred_batch = pred_frame.unsqueeze(0)
            
            try:
                score = metric(pred_batch)
                if isinstance(score, torch.Tensor):
                    score_val = score.item()
                else:
                    score_val = float(score)
                musiq_vals.append(score_val)
            except Exception:
                continue
    
    if not musiq_vals:
        return float("nan")
    return float(np.mean(musiq_vals))


def compute_video_metrics(
    real_videos: List[torch.Tensor],
    fake_videos: List[torch.Tensor],
    device: Union[torch.device, str],
    data_range: float = 1.0,
    compute_fvd: bool = True,
    fvd_batch_size: int = 2,
    max_fvd_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute PSNR, SSIM (using pyiqa), and optionally FVD over two lists
    of videos. Each element is a tensor (C, T, H, W) in [0, data_range].

    Returns dict with keys: "psnr", "ssim" (or nan), "fvd" (if compute_fvd and available).
    """
    device = torch.device(device) if isinstance(device, str) else device
    if not real_videos or not fake_videos:
        return {"psnr": float("nan"), "ssim": float("nan"), "fvd": float("nan")}

    n = min(len(real_videos), len(fake_videos))
    real_videos = real_videos[:n]
    fake_videos = fake_videos[:n]

    psnr_vals = []
    ssim_vals = []
    for r, f in zip(real_videos, fake_videos):
        t = min(r.shape[1], f.shape[1])
        r, f = r[:, :t], f[:, :t]
        # Check spatial size (H, W): caller (e.g. inference.py) should align before calling.
        if r.shape[2:] != f.shape[2:]:
            continue  # skip this pair
        p = compute_psnr(f, r, device=device, data_range=data_range)
        s = compute_ssim_video(f, r, device=device, data_range=data_range)
        # 只有非 nan 的值才用于计算平均值
        if p == p:  # Check for nan
            psnr_vals.append(p)
        if s == s:  # Check for nan
            ssim_vals.append(s)

    out = {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
    }

    if compute_fvd and _compute_fvd is not None and real_videos and fake_videos:
        max_items = n if max_fvd_samples is None else min(n, max_fvd_samples)
        if max_items < 2:
            out["fvd"] = float("nan")  # FVD requires at least 2 samples for covariance
        else:
            # Only include pairs with matching spatial size (caller should align, e.g. inference.py).
            real_aligned, pred_aligned = [], []
            for r, f in zip(real_videos[:max_items], fake_videos[:max_items]):
                t = min(r.shape[1], f.shape[1])
                r, f = r[:, :t], f[:, :t]
                if r.shape[2:] != f.shape[2:]:
                    continue
                real_aligned.append(r)
                pred_aligned.append(f)
            if len(real_aligned) < 2:
                out["fvd"] = float("nan")
            else:
                y_true = torch.stack(real_aligned, dim=0).to(device=device, dtype=torch.float32)
                y_pred = torch.stack(pred_aligned, dim=0).to(device=device, dtype=torch.float32)
                y_true = (y_true * 2.0 - 1.0)
                y_pred = (y_pred * 2.0 - 1.0)
                try:
                    out["fvd"] = float(
                        _compute_fvd(y_true, y_pred, max_items, device, fvd_batch_size)
                    )
                except Exception:
                    out["fvd"] = float("nan")
    else:
        out["fvd"] = float("nan")

    return out


def compute_video_metrics_detailed(
    real_videos: List[torch.Tensor],
    fake_videos: List[torch.Tensor],
    device: Union[torch.device, str],
    data_range: float = 1.0,
    compute_fvd: bool = True,
    fvd_batch_size: int = 2,
    max_fvd_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Like compute_video_metrics but returns per-video PSNR/SSIM/LPIPS/MUSIQ and a summary.

    Returns dict with:
      - "per_video": list of {"psnr": float, "ssim": float, "lpips": float, "musiq": float} for each pair (nan if skipped)
      - "summary": {"num_videos_with_valid_metrics": int, "mean_psnr": float, "mean_ssim": float, "mean_lpips": float, "mean_musiq": float, "fvd": float}
    """
    device = torch.device(device) if isinstance(device, str) else device
    nan_val = float("nan")
    if not real_videos or not fake_videos:
        return {
            "per_video": [],
            "summary": {
                "num_videos_with_valid_metrics": 0,
                "mean_psnr": nan_val,
                "mean_ssim": nan_val,
                "mean_lpips": nan_val,
                "mean_musiq": nan_val,
                "fvd": nan_val,
            },
        }

    n = min(len(real_videos), len(fake_videos))
    real_videos = real_videos[:n]
    fake_videos = fake_videos[:n]

    per_video = []
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    musiq_vals = []
    for r, f in zip(real_videos, fake_videos):
        t = min(r.shape[1], f.shape[1])
        r, f = r[:, :t], f[:, :t]
        if r.shape[2:] != f.shape[2:]:
            per_video.append({"psnr": nan_val, "ssim": nan_val, "lpips": nan_val, "musiq": nan_val})
            continue
        p = compute_psnr(f, r, device=device, data_range=data_range)
        s = compute_ssim_video(f, r, device=device, data_range=data_range)
        l = compute_lpips_video(f, r, device=device)
        m = compute_musiq_video(f, device=device)  # MUSIQ 只需要 pred，不需要 GT
        per_video.append({"psnr": float(p), "ssim": float(s), "lpips": float(l), "musiq": float(m)})
        # 只有非 nan 的值才用于计算平均值
        if p == p:  # Check for nan
            psnr_vals.append(p)
        if s == s:  # Check for nan
            ssim_vals.append(s)
        if l == l:  # Check for nan
            lpips_vals.append(l)
        if m == m:  # Check for nan
            musiq_vals.append(m)

    num_valid = len(psnr_vals)
    mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else nan_val
    mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else nan_val
    mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else nan_val
    mean_musiq = float(np.mean(musiq_vals)) if musiq_vals else nan_val

    fvd_val = nan_val
    # 调试信息：检查 FVD 计算条件
    import sys
    if not compute_fvd:
        print("[FVD] Skipped: compute_fvd=False", file=sys.stderr)
    elif _compute_fvd is None:
        print("[FVD] Skipped: _compute_fvd is None (FVD module not available)", file=sys.stderr)
    elif not real_videos or not fake_videos:
        print(f"[FVD] Skipped: real_videos={len(real_videos) if real_videos else 0}, fake_videos={len(fake_videos) if fake_videos else 0}", file=sys.stderr)
    
    if compute_fvd and _compute_fvd is not None and real_videos and fake_videos:
        max_items = n if max_fvd_samples is None else min(n, max_fvd_samples)
        print(f"[FVD] max_items={max_items}, n={n}, max_fvd_samples={max_fvd_samples}", file=sys.stderr)
        if max_items >= 2:
            # 收集所有视频的尺寸信息用于日志
            video_shapes = []
            for r, f in zip(real_videos[:max_items], fake_videos[:max_items]):
                r_shape = (r.shape[0], r.shape[1], r.shape[2], r.shape[3])  # C, T, H, W
                f_shape = (f.shape[0], f.shape[1], f.shape[2], f.shape[3])  # C, T, H, W
                video_shapes.append({"real": r_shape, "pred": f_shape})
            
            # 检查所有视频的分辨率和帧数是否相同
            all_same_resolution = True
            all_same_frames = True
            target_shape = None
            
            if video_shapes:
                first_real_shape = video_shapes[0]["real"]
                first_pred_shape = video_shapes[0]["pred"]
                target_shape = first_real_shape  # 使用第一个 real 视频的尺寸作为目标
                
                for i, shapes in enumerate(video_shapes):
                    r_shape, f_shape = shapes["real"], shapes["pred"]
                    # 检查分辨率 (H, W)
                    if r_shape[2:] != target_shape[2:] or f_shape[2:] != target_shape[2:]:
                        all_same_resolution = False
                    # 检查帧数 (T)
                    if r_shape[1] != target_shape[1] or f_shape[1] != target_shape[1]:
                        all_same_frames = False
            
            # 输出日志信息
            print(f"[FVD] Checking video dimensions for {max_items} videos...", file=sys.stderr)
            if all_same_resolution and all_same_frames:
                print(f"[FVD] All videos have same resolution and frame count: {target_shape[2]}x{target_shape[3]} @ {target_shape[1]} frames", file=sys.stderr)
            else:
                print(f"[FVD] Videos have different dimensions, alignment needed:", file=sys.stderr)
                print(f"[FVD]   Target shape: C={target_shape[0]}, T={target_shape[1]}, H={target_shape[2]}, W={target_shape[3]}", file=sys.stderr)
                if not all_same_resolution:
                    print(f"[FVD]   Resolution mismatch detected - will resize to {target_shape[2]}x{target_shape[3]}", file=sys.stderr)
                if not all_same_frames:
                    print(f"[FVD]   Frame count mismatch detected - will align to {target_shape[1]} frames", file=sys.stderr)
            
            # 对齐所有视频到统一尺寸：使用第一个 real 视频的尺寸作为目标
            real_aligned, pred_aligned = [], []
            for r, f in zip(real_videos[:max_items], fake_videos[:max_items]):
                # 对齐帧数：截断到目标帧数
                target_frames = target_shape[1]
                r_aligned = r[:, :target_frames] if r.shape[1] >= target_frames else r
                f_aligned = f[:, :target_frames] if f.shape[1] >= target_frames else f
                
                # 如果帧数不足，用最后一帧填充（但通常应该截断）
                if r_aligned.shape[1] < target_frames:
                    last_frame = r_aligned[:, -1:, :, :]
                    padding = last_frame.repeat(1, target_frames - r_aligned.shape[1], 1, 1)
                    r_aligned = torch.cat([r_aligned, padding], dim=1)
                if f_aligned.shape[1] < target_frames:
                    last_frame = f_aligned[:, -1:, :, :]
                    padding = last_frame.repeat(1, target_frames - f_aligned.shape[1], 1, 1)
                    f_aligned = torch.cat([f_aligned, padding], dim=1)
                
                # 对齐分辨率：resize 到目标分辨率
                target_h, target_w = target_shape[2], target_shape[3]
                if r_aligned.shape[2:] != (target_h, target_w):
                    r_aligned = torch.nn.functional.interpolate(
                        r_aligned, size=(target_h, target_w), mode='bilinear', align_corners=False
                    )
                if f_aligned.shape[2:] != (target_h, target_w):
                    f_aligned = torch.nn.functional.interpolate(
                        f_aligned, size=(target_h, target_w), mode='bilinear', align_corners=False
                    )
                
                real_aligned.append(r_aligned)
                pred_aligned.append(f_aligned)
            
            if len(real_aligned) >= 2:
                # 转换为 [-1, 1] 范围（FVD 期望），并移动到 device
                real_aligned_norm = [(v * 2.0 - 1.0).to(device=device, dtype=torch.float32) for v in real_aligned]
                pred_aligned_norm = [(v * 2.0 - 1.0).to(device=device, dtype=torch.float32) for v in pred_aligned]
                
                # Stack: (num_videos, C, T, H, W) - 用于传递给 compute_fvd
                # compute_fvd 内部会将其转换为 list 格式
                y_true = torch.stack(real_aligned_norm, dim=0)
                y_pred = torch.stack(pred_aligned_norm, dim=0)
                
                print(f"[FVD] Stacked tensor shapes: y_true={y_true.shape}, y_pred={y_pred.shape}", file=sys.stderr)
                print(f"[FVD]   Format: (num_videos={y_true.shape[0]}, channels={y_true.shape[1]}, frames={y_true.shape[2]}, height={y_true.shape[3]}, width={y_true.shape[4]})", file=sys.stderr)
                
                try:
                    fvd_val = float(
                        _compute_fvd(y_true, y_pred, max_items, device, fvd_batch_size)
                    )
                    print(f"[FVD] Computed FVD: {fvd_val:.4f}", file=sys.stderr)
                except Exception as e:
                    print(f"[FVD] Error computing FVD: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    pass

    return {
        "per_video": per_video,
        "summary": {
            "num_videos_with_valid_metrics": num_valid,
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "mean_lpips": mean_lpips,
            "mean_musiq": mean_musiq,
            "fvd": fvd_val,
        },
    }


def gather_video_pairs_to_rank0(
    rank: int,
    world_size: int,
    real_list: List[torch.Tensor],
    pred_list: List[torch.Tensor],
    id_list: Optional[List[Any]] = None,
) -> tuple:
    """
    In a multi-GPU run, gather each rank's (real_list, pred_list[, id_list]) to rank 0.
    Each list contains tensors of shape (C, T, H, W); can be on CPU to save GPU memory.

    Returns:
        If id_list is None: (real_all, pred_all) on rank 0; (None, None) on non-zero ranks.
        If id_list is not None: (real_all, pred_all, ids_all) on rank 0; (None, None, None) on non-zero ranks.
    """
    if world_size <= 1:
        if id_list is not None:
            return (real_list, pred_list, id_list)
        return (real_list, pred_list)

    try:
        import torch.distributed as dist
    except ImportError:
        if id_list is not None:
            return (real_list, pred_list, id_list)
        return (real_list, pred_list)

    if not dist.is_initialized():
        if id_list is not None:
            return (real_list, pred_list, id_list)
        return (real_list, pred_list)

    payload = [real_list, pred_list] if id_list is None else [real_list, pred_list, id_list]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, payload)

    if rank != 0:
        if id_list is not None:
            return (None, None, None)
        return (None, None)

    real_all = []
    pred_all = []
    ids_all = [] if id_list is not None else None
    for item in gathered:
        if id_list is None:
            r_list, p_list = item
            real_all.extend(r_list)
            pred_all.extend(p_list)
        else:
            r_list, p_list, i_list = item
            real_all.extend(r_list)
            pred_all.extend(p_list)
            ids_all.extend(i_list)
    if id_list is not None:
        return (real_all, pred_all, ids_all)
    return (real_all, pred_all)
