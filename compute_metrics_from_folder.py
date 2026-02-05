#!/usr/bin/env python3
"""
独立脚本：基于文件夹中的 *_gt.mp4 和 *_pred.mp4 计算视频指标（PSNR, SSIM, LPIPS, MUSIQ, FVD）
并输出每个视频的分辨率信息。
"""
import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from PIL import Image

from diffsynth.utils.data import VideoData
from diffsynth.metrics.video_metrics import (
    video_to_tensor,
    compute_psnr,
    compute_ssim_video,
    compute_musiq_video,
    compute_video_metrics_detailed,
)


def get_video_info(video_path: str) -> Dict[str, int]:
    """获取视频信息：帧数、高度、宽度"""
    loader = VideoData(video_path)
    frames = loader.raw_data()
    if hasattr(loader.data, "reader") and loader.data.reader is not None:
        loader.data.reader.close()
    
    if not frames:
        return {"num_frames": 0, "height": 0, "width": 0}
    
    first_frame = frames[0]
    if isinstance(first_frame, Image.Image):
        width, height = first_frame.size  # PIL: (width, height)
    else:
        # numpy array: (height, width) or (height, width, channels)
        height, width = first_frame.shape[:2]
    
    return {
        "num_frames": len(frames),
        "height": height,
        "width": width,
    }


def find_video_pairs(folder: str) -> List[Tuple[str, str, str]]:
    """
    在文件夹中找到所有 (base_name, gt_path, pred_path) 配对
    返回: [(base_name, gt_path, pred_path), ...]
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder}")
    
    gt_files = sorted(folder_path.glob("*_gt.mp4"))
    pairs = []
    
    for gt_path in gt_files:
        base_name = gt_path.stem.replace("_gt", "")
        pred_path = folder_path / f"{base_name}_pred.mp4"
        
        if pred_path.exists():
            pairs.append((base_name, str(gt_path), str(pred_path)))
        else:
            print(f"[WARN] GT file {gt_path.name} exists but pred file {pred_path.name} not found, skipping")
    
    return pairs


def load_and_align_videos(gt_path: str, pred_path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    加载 GT 和 pred 视频，对齐尺寸，返回 tensor 和分辨率信息
    返回: (gt_tensor, pred_tensor, info_dict)
    """
    gt_loader = VideoData(gt_path)
    pred_loader = VideoData(pred_path)
    
    gt_frames = gt_loader.raw_data()
    pred_frames = pred_loader.raw_data()
    
    if hasattr(gt_loader.data, "reader") and gt_loader.data.reader is not None:
        gt_loader.data.reader.close()
    if hasattr(pred_loader.data, "reader") and pred_loader.data.reader is not None:
        pred_loader.data.reader.close()
    
    # 获取原始分辨率
    if isinstance(gt_frames[0], Image.Image):
        gt_w, gt_h = gt_frames[0].size  # PIL: (width, height)
    else:
        gt_h, gt_w = gt_frames[0].shape[:2]
    
    if isinstance(pred_frames[0], Image.Image):
        pred_w, pred_h = pred_frames[0].size  # PIL: (width, height)
    else:
        pred_h, pred_w = pred_frames[0].shape[:2]
    
    # 对齐帧数
    min_frames = min(len(gt_frames), len(pred_frames))
    gt_frames = gt_frames[:min_frames]
    pred_frames = pred_frames[:min_frames]
    
    # 对齐空间尺寸：resize pred 到 GT 尺寸
    if isinstance(pred_frames[0], Image.Image):
        pred_frames_aligned = [f.resize((gt_w, gt_h), Image.BICUBIC) for f in pred_frames]
    else:
        # numpy array: 需要转换为 PIL 或使用 cv2
        # 这里统一转换为 PIL Image 处理
        pred_frames_aligned = []
        for f in pred_frames:
            if isinstance(f, Image.Image):
                pred_frames_aligned.append(f.resize((gt_w, gt_h), Image.BICUBIC))
            else:
                # numpy array -> PIL Image
                pil_img = Image.fromarray(f.astype('uint8') if f.dtype != 'uint8' else f)
                pred_frames_aligned.append(pil_img.resize((gt_w, gt_h), Image.BICUBIC))
    
    gt_tensor = video_to_tensor(gt_frames)
    pred_tensor = video_to_tensor(pred_frames_aligned)
    
    info = {
        "gt_resolution": f"{gt_h}x{gt_w}",
        "pred_original_resolution": f"{pred_h}x{pred_w}",
        "pred_aligned_resolution": f"{gt_h}x{gt_w}",
        "num_frames": min_frames,
    }
    
    return gt_tensor, pred_tensor, info


def main():
    parser = argparse.ArgumentParser(description="计算文件夹中视频对的指标")
    parser.add_argument("folder", type=str, help="包含 *_gt.mp4 和 *_pred.mp4 的文件夹路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径（默认：folder/metrics.json）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--fvd_batch_size", type=int, default=4, help="FVD 计算的 batch size")
    
    args = parser.parse_args()
    
    folder = args.folder
    output_path = args.output if args.output else os.path.join(folder, "metrics.json")
    device = torch.device(args.device)
    
    print(f"[INFO] Scanning folder: {folder}")
    pairs = find_video_pairs(folder)
    
    if not pairs:
        print("[ERROR] No video pairs found!")
        return
    
    print(f"[INFO] Found {len(pairs)} video pairs")
    
    # 收集所有视频对和分辨率信息
    real_videos = []
    fake_videos = []
    video_info_list = []
    
    for base_name, gt_path, pred_path in pairs:
        print(f"\n[INFO] Processing: {base_name}")
        
        # 获取分辨率信息
        gt_info = get_video_info(gt_path)
        pred_info = get_video_info(pred_path)
        
        print(f"  GT:   {gt_info['num_frames']} frames, {gt_info['height']}x{gt_info['width']}")
        print(f"  Pred:  {pred_info['num_frames']} frames, {pred_info['height']}x{pred_info['width']}")
        
        # 加载并对齐视频
        try:
            gt_tensor, pred_tensor, align_info = load_and_align_videos(gt_path, pred_path)
            real_videos.append(gt_tensor.cpu())
            fake_videos.append(pred_tensor.cpu())
            
            video_info_list.append({
                "clip_id": base_name,
                "gt_path": gt_path,
                "pred_path": pred_path,
                "gt_resolution": align_info["gt_resolution"],
                "pred_original_resolution": align_info["pred_original_resolution"],
                "pred_aligned_resolution": align_info["pred_aligned_resolution"],
                "num_frames": align_info["num_frames"],
            })
            print(f"  Aligned: {align_info['num_frames']} frames, {align_info['pred_aligned_resolution']}")
        except Exception as e:
            print(f"  [ERROR] Failed to load/align: {e}")
            continue
    
    if not real_videos or not fake_videos:
        print("[ERROR] No valid video pairs after loading!")
        return
    
    print(f"\n[INFO] Computing metrics on {len(real_videos)} video pairs...")
    
    # 计算详细指标（包括 pred 的 MUSIQ）
    detailed = compute_video_metrics_detailed(
        real_videos,
        fake_videos,
        device=device,
        data_range=1.0,
        compute_fvd=True,
        fvd_batch_size=args.fvd_batch_size,
        max_fvd_samples=None,
    )
    
    # 计算 GT 的 MUSIQ（no-reference 指标，GT 也可以计算）
    print(f"[INFO] Computing MUSIQ for GT videos...")
    gt_musiq_vals = []
    for i, gt_tensor in enumerate(real_videos):
        try:
            gt_musiq = compute_musiq_video(gt_tensor, device=device)
            gt_musiq_vals.append(gt_musiq)
            if gt_musiq == gt_musiq:  # Check for nan
                print(f"  [{i+1}/{len(real_videos)}] GT MUSIQ: {gt_musiq:.4f}")
            else:
                print(f"  [{i+1}/{len(real_videos)}] GT MUSIQ: NaN")
        except Exception as e:
            print(f"  [{i+1}/{len(real_videos)}] Failed to compute GT MUSIQ: {e}")
            gt_musiq_vals.append(float("nan"))
    
    mean_gt_musiq = float(np.mean([v for v in gt_musiq_vals if v == v])) if any(v == v for v in gt_musiq_vals) else float("nan")
    print(f"[INFO] Mean GT MUSIQ: {mean_gt_musiq:.4f}" if mean_gt_musiq == mean_gt_musiq else "[INFO] Mean GT MUSIQ: NaN")
    
    # 构建输出：合并 per_video 指标和分辨率信息
    per_video_output = []
    for i, pv in enumerate(detailed["per_video"]):
        entry = {
            "clip_id": video_info_list[i]["clip_id"] if i < len(video_info_list) else f"video_{i}",
            "psnr": pv["psnr"],
            "ssim": pv["ssim"],
            "lpips": pv.get("lpips", float("nan")),
            "musiq_pred": pv.get("musiq", float("nan")),
            "musiq_gt": gt_musiq_vals[i] if i < len(gt_musiq_vals) else float("nan"),
        }
        if i < len(video_info_list):
            entry.update({
                "gt_resolution": video_info_list[i]["gt_resolution"],
                "pred_original_resolution": video_info_list[i]["pred_original_resolution"],
                "pred_aligned_resolution": video_info_list[i]["pred_aligned_resolution"],
                "num_frames": video_info_list[i]["num_frames"],
            })
        per_video_output.append(entry)
    
    # 从 per_video_output 中提取 musiq_pred 计算平均值
    pred_musiq_vals = [e["musiq_pred"] for e in per_video_output if e["musiq_pred"] == e["musiq_pred"]]
    mean_pred_musiq = float(np.mean(pred_musiq_vals)) if pred_musiq_vals else float("nan")
    print(f"[INFO] Mean Pred MUSIQ: {mean_pred_musiq:.4f}" if mean_pred_musiq == mean_pred_musiq else "[INFO] Mean Pred MUSIQ: NaN")
    
    # 更新 summary，添加 pred 和 GT MUSIQ
    summary = detailed["summary"].copy()
    summary["mean_musiq_pred"] = mean_pred_musiq
    summary["mean_musiq_gt"] = mean_gt_musiq
    
    output = {
        "per_video": per_video_output,
        "summary": summary,
    }
    
    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # 打印摘要（使用更新后的 summary，包含 GT MUSIQ）
    print(f"\n[RESULTS]")
    print(f"  Valid video pairs: {summary['num_videos_with_valid_metrics']}")
    
    # 统一处理所有指标：检查 NaN
    psnr_val = summary['mean_psnr']
    ssim_val = summary['mean_ssim']
    lpips_val = summary.get('mean_lpips', float('nan'))
    musiq_pred_val = summary.get('mean_musiq_pred', summary.get('mean_musiq', float('nan')))
    musiq_gt_val = summary.get('mean_musiq_gt', float('nan'))
    fvd_val = summary['fvd']
    
    psnr_str = f"{psnr_val:.4f}" if psnr_val == psnr_val else "NaN"
    ssim_str = f"{ssim_val:.4f}" if ssim_val == ssim_val else "NaN"
    lpips_str = f"{lpips_val:.4f}" if lpips_val == lpips_val else "NaN"
    musiq_pred_str = f"{musiq_pred_val:.4f}" if musiq_pred_val == musiq_pred_val else "NaN"
    musiq_gt_str = f"{musiq_gt_val:.4f}" if musiq_gt_val == musiq_gt_val else "NaN"
    fvd_str = f"{fvd_val:.4f}" if fvd_val == fvd_val else "NaN"
    
    print(f"  Mean PSNR: {psnr_str}")
    print(f"  Mean SSIM: {ssim_str}")
    print(f"  Mean LPIPS: {lpips_str}")
    print(f"  Mean MUSIQ (Pred): {musiq_pred_str}")
    print(f"  Mean MUSIQ (GT): {musiq_gt_str}")
    print(f"  FVD: {fvd_str}")
    print(f"\n[INFO] Results saved to: {output_path}")
    
    # 打印每个视频的分辨率和指标
    print(f"\n[VIDEO DETAILS]")
    for i, vinfo in enumerate(video_info_list):
        print(f"  {vinfo['clip_id']}:")
        print(f"    Resolution: GT={vinfo['gt_resolution']}, Pred={vinfo['pred_original_resolution']} -> {vinfo['pred_aligned_resolution']} ({vinfo['num_frames']} frames)")
        if i < len(per_video_output):
            pv = per_video_output[i]
            p_psnr = pv['psnr']
            p_ssim = pv['ssim']
            p_lpips = pv.get('lpips', float('nan'))
            p_musiq_pred = pv.get('musiq_pred', float('nan'))
            p_musiq_gt = pv.get('musiq_gt', float('nan'))
            psnr_str = f"{p_psnr:.4f}" if p_psnr == p_psnr else "NaN"
            ssim_str = f"{p_ssim:.4f}" if p_ssim == p_ssim else "NaN"
            lpips_str = f"{p_lpips:.4f}" if p_lpips == p_lpips else "NaN"
            musiq_pred_str = f"{p_musiq_pred:.4f}" if p_musiq_pred == p_musiq_pred else "NaN"
            musiq_gt_str = f"{p_musiq_gt:.4f}" if p_musiq_gt == p_musiq_gt else "NaN"
            print(f"    Metrics: PSNR={psnr_str}, SSIM={ssim_str}, LPIPS={lpips_str}")
            print(f"    MUSIQ: Pred={musiq_pred_str}, GT={musiq_gt_str}")


if __name__ == "__main__":
    main()
