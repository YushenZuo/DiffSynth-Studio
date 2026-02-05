"""
Generic video quality metrics (PSNR, SSIM, FVD) for use in both training validation
and inference. Supports multi-GPU: gather (gt, pred) pairs from all ranks to rank 0
and compute metrics on rank 0.
"""
import math
import os
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch
from PIL import Image

try:
    from torchmetrics.functional.image import structural_similarity_index_measure as _ssim_fn  # type: ignore[import]
except ImportError:
    _ssim_fn = None

try:
    from diffsynth.metrics.fvd_metric import compute_fvd as _compute_fvd  # type: ignore[import]
except ImportError:
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


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """PSNR between two videos (C, T, H, W)."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def compute_ssim_video(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """SSIM over video (C, T, H, W) via per-frame SSIM then mean. Requires torchmetrics."""
    if _ssim_fn is None:
        return float("nan")
    pred_frames = pred.permute(1, 0, 2, 3)
    target_frames = target.permute(1, 0, 2, 3)
    with torch.no_grad():
        return float(_ssim_fn(pred_frames, target_frames, data_range=data_range).item())


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
    Compute PSNR, SSIM (if torchmetrics available), and optionally FVD over two lists
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
        psnr_vals.append(compute_psnr(f, r, data_range=data_range))
        if _ssim_fn is not None:
            ssim_vals.append(compute_ssim_video(f, r, data_range=data_range))

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
    Like compute_video_metrics but returns per-video PSNR/SSIM and a summary.

    Returns dict with:
      - "per_video": list of {"psnr": float, "ssim": float} for each pair (nan if skipped)
      - "summary": {"num_videos_with_valid_metrics": int, "mean_psnr": float, "mean_ssim": float, "fvd": float}
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
                "fvd": nan_val,
            },
        }

    n = min(len(real_videos), len(fake_videos))
    real_videos = real_videos[:n]
    fake_videos = fake_videos[:n]

    per_video = []
    psnr_vals = []
    ssim_vals = []
    for r, f in zip(real_videos, fake_videos):
        t = min(r.shape[1], f.shape[1])
        r, f = r[:, :t], f[:, :t]
        if r.shape[2:] != f.shape[2:]:
            per_video.append({"psnr": nan_val, "ssim": nan_val})
            continue
        p = compute_psnr(f, r, data_range=data_range)
        s = compute_ssim_video(f, r, data_range=data_range) if _ssim_fn is not None else nan_val
        per_video.append({"psnr": float(p), "ssim": float(s)})
        psnr_vals.append(p)
        ssim_vals.append(s)

    num_valid = len(psnr_vals)
    mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else nan_val
    mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else nan_val

    fvd_val = nan_val
    if compute_fvd and _compute_fvd is not None and real_videos and fake_videos:
        max_items = n if max_fvd_samples is None else min(n, max_fvd_samples)
        if max_items >= 2:
            real_aligned, pred_aligned = [], []
            for r, f in zip(real_videos[:max_items], fake_videos[:max_items]):
                t = min(r.shape[1], f.shape[1])
                r, f = r[:, :t], f[:, :t]
                if r.shape[2:] != f.shape[2:]:
                    continue
                real_aligned.append(r)
                pred_aligned.append(f)
            if len(real_aligned) >= 2:
                y_true = torch.stack(real_aligned, dim=0).to(device=device, dtype=torch.float32)
                y_pred = torch.stack(pred_aligned, dim=0).to(device=device, dtype=torch.float32)
                y_true = (y_true * 2.0 - 1.0)
                y_pred = (y_pred * 2.0 - 1.0)
                try:
                    fvd_val = float(
                        _compute_fvd(y_true, y_pred, max_items, device, fvd_batch_size)
                    )
                except Exception:
                    pass

    return {
        "per_video": per_video,
        "summary": {
            "num_videos_with_valid_metrics": num_valid,
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
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
