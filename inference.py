import argparse, os, shutil, torch
import torch.distributed as dist
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.metrics.video_metrics import (
    video_to_tensor,
    compute_video_metrics_detailed,
    gather_video_pairs_to_rank0,
)


MODEL_CONFIGS = [
    "diffusion_pytorch_model*.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
]


def build_val_subset_df(df, val_subset=0, val_ids_file=None):
    """
    Restrict df to a validation subset in the same way as training validation:
    - If val_ids_file: load val_ids from JSON, keep rows whose clip_id is in that list (first occurrence per id).
    - Else if val_subset > 0: first N unique clip_id in CSV order (same as training get_stable_id iteration).
    """
    if val_subset <= 0 and not val_ids_file:
        return df
    if val_ids_file and os.path.isfile(val_ids_file):
        import json
        with open(val_ids_file, "r") as f:
            saved = json.load(f)
        raw = saved.get("val_ids", saved.get("val_indices", []))
        val_ids = [tuple(x) if isinstance(x, list) else x for x in raw]
        # Normalize to comparable clip_id (training may store (stem, parent) or scalar)
        def norm_id(x):
            if isinstance(x, (list, tuple)):
                return str(x[0]) if len(x) else ""
            return str(x)
        allowed = {norm_id(x) for x in val_ids}
        if not allowed:
            return df
        # First occurrence of each allowed clip_id, order by val_ids
        seen = set()
        ordered = []
        for vid in val_ids:
            cid = norm_id(vid)
            if cid in allowed and cid not in seen:
                seen.add(cid)
                idx = df[df["clip_id"].astype(str) == cid].index
                if len(idx):
                    ordered.append(idx[0])
        if ordered:
            df = df.loc[ordered].reset_index(drop=True)
        return df
    # First N unique clip_id in dataframe order (same logic as training validation)
    seen = set()
    rows = []
    for _, row in df.iterrows():
        cid = row["clip_id"]
        cid_key = tuple(cid) if isinstance(cid, (list, tuple)) else cid
        if cid_key not in seen:
            seen.add(cid_key)
            rows.append(row)
        if len(rows) >= val_subset:
            break
    return pd.DataFrame(rows).reset_index(drop=True) if rows else df


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Inference for Wan2.2 Animate with fixed input image.")
    parser.add_argument(
        "--test_csv",
        type=str,
        default="./data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv",
        help="Path to the testing metadata CSV.",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="/mnt/beegfs/mingyang/Human-Replacement/data/ID-Library/example/full-body_view_1.png",
        help="Path to the single reference identity image.",
    )
    parser.add_argument(
        "--input_ref_key",
        type=str,
        default="inference_ref_image",
        help="keys: inference_ref_image, animate_ref_identity_image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/demo",
        help="Output directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/beegfs/mingyang/Human-Replacement/models/train/v0.1/20260111_110445-Arcface_Module_DiT_LoRA128-Data_20260103_11k-Ref_Segment/step-4000.safetensors",
        help="Adapter checkpoint to load.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "base_wo_relighting", "ft_animate-adapter", "ft_dit_lora", "ft_animate-adapter_dit_lora", "arcface", "arcface_dit_lora"],
        default="arcface_dit_lora",
        help="Select base (skip adapter) or adapter (load adapter) inference flow.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames to generate per sample.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Inference steps per frame.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--prefix_video_name",
        type=str,
        default="Wan-Animate",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=0,
        help="If > 0, randomly sample this many rows from the CSV.",
    )
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="Compute PSNR / SSIM / FVD (with multi-GPU gather to rank 0) and save to output_dir/metrics.json.",
    )
    parser.add_argument(
        "--metrics_max_videos",
        type=int,
        default=200,
        help="Max number of videos to use for metrics when --compute_metrics (cap total across ranks). Default 200.",
    )
    parser.add_argument(
        "--val_subset",
        type=int,
        default=0,
        help="If > 0, use a validation subset of this size (first N unique clip_id in CSV order, same logic as training val). Enables comparable metrics to training validation.",
    )
    parser.add_argument(
        "--val_ids_file",
        type=str,
        default=None,
        help="Optional path to val_ids.json from training (output_path/val/val_ids.json). Use exactly the same clip ids as training validation.",
    )
    return parser.parse_args()


def build_pipeline(args, device):
    if args.mode in {"base", "base_wo_relighting", "ft_animate-adapter", "ft_dit_lora", "ft_animate-adapter_dit_lora"}:
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern=pattern) for pattern in MODEL_CONFIGS],
        )
    elif args.mode in {"arcface", "arcface_dit_lora"}:
        from diffsynth.pipelines.wan_video_face import WanVideoPipeline, ModelConfig
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern=pattern) for pattern in MODEL_CONFIGS],
            face_model_path="/mnt/beegfs/mingyang/Human-Replacement",
        )
    else:
        raise ValueError(f"[ERROR] Unsupported mode: {args.mode}")
    return pipe


def load_adapters(args, pipe, relighting_lora_state_dict):
    if args.mode == "base":
        print("[INFO] Base mode selected;")
        pipe.load_lora(pipe.dit, state_dict=relighting_lora_state_dict)
    elif args.mode == "base_wo_relighting":
        print("[INFO] Base mode selected; skipping adapter load.")
    elif args.mode == "ft_animate-adapter":
        print(f"[INFO] Loading Wan-Animate Adapter: {args.checkpoint}")
        state_dict = load_state_dict(args.checkpoint)
        pipe.animate_adapter.load_state_dict(state_dict, strict=False)
        pipe.load_lora(pipe.dit, state_dict=relighting_lora_state_dict)
    elif args.mode == "ft_dit_lora":
        print(f"[INFO] Loading DiT (LoRA): {args.checkpoint}")
        state_dict = load_state_dict(args.checkpoint)
        pipe_dit_lora_state = {k[len("pipe.dit."):]: v for k, v in state_dict.items() if k.startswith("pipe.dit.")}
        pipe.load_lora(pipe.dit, state_dict=pipe_dit_lora_state)
    elif args.mode == "ft_animate-adapter_dit_lora":
        print(f"[INFO] Loading Wan-Animate Adapter and DiT (LoRA): {args.checkpoint}")
        state_dict = load_state_dict(args.checkpoint)
        pipe_dit_lora_state = {k[len("pipe.dit."):]: v for k, v in state_dict.items() if k.startswith("pipe.dit.")}
        adapter_keys = ("face_encoder", "face_adapter", "motion_encoder", "pose_patch_embedding")
        animate_adapter_state = {k: v for k, v in state_dict.items() if k.startswith(adapter_keys)}
        pipe.animate_adapter.load_state_dict(animate_adapter_state, strict=False)
        pipe.load_lora(pipe.dit, state_dict=pipe_dit_lora_state)
    elif args.mode == "arcface":
        state_dict = load_state_dict(args.checkpoint)
        animate_arcface_adapter_state = {k[len("pipe.animate_arcface_adapter."):]: v for k, v in state_dict.items() if k.startswith("pipe.animate_arcface_adapter.")}
        face_encoder_state = {k[len("pipe.face_encoder."):]: v for k, v in state_dict.items() if k.startswith("pipe.face_encoder.")}
        pipe.animate_arcface_adapter.load_state_dict(animate_arcface_adapter_state, strict=False)
        pipe.face_encoder.load_state_dict(face_encoder_state, strict=False)
    elif args.mode == "arcface_dit_lora":
        state_dict = load_state_dict(args.checkpoint)
        animate_arcface_adapter_state = {k[len("pipe.animate_arcface_adapter."):]: v for k, v in state_dict.items() if k.startswith("pipe.animate_arcface_adapter.")}
        face_encoder_state = {k[len("pipe.face_encoder."):]: v for k, v in state_dict.items() if k.startswith("pipe.face_encoder.")}
        pipe_dit_lora_state = {k[len("pipe.dit."):]: v for k, v in state_dict.items() if k.startswith("pipe.dit.")}
        pipe.animate_arcface_adapter.load_state_dict(animate_arcface_adapter_state, strict=False)
        pipe.face_encoder.load_state_dict(face_encoder_state, strict=False)
        pipe.load_lora(pipe.dit, state_dict=pipe_dit_lora_state)


def get_video_size(video_path):
    video_loader = VideoData(video_path)
    video_width, video_height = video_loader[0].size
    if hasattr(video_loader.data, "reader"):
        video_loader.data.reader.close()
    del video_loader
    return video_width, video_height


def get_ref_image_path(args, row, has_csv_ref_image, input_ref_key=None, rank=0):
    ref_image_path = args.input_image
    if has_csv_ref_image:
        csv_ref_image = row.get(input_ref_key)
        if rank == 0:
            print(f"Reference Image: {csv_ref_image}")
        if pd.notna(csv_ref_image) and str(csv_ref_image).strip() != "":
            ref_image_path = csv_ref_image
    return ref_image_path


def main():
    args = parse_args()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[INFO] World Size: {world_size}, Output Dir: {args.output_dir}")
    if world_size > 1:
        dist.barrier()

    ref_name = Path(args.input_image).stem

    pipe = build_pipeline(args, device=device)

    relighting_lora_path = "models/Wan-AI/Wan2.2-Animate-14B/relighting_lora.ckpt"
    relighting_lora_state_dict = load_state_dict(
        relighting_lora_path,
        torch_dtype=torch.bfloat16,
        device=device,
    )["state_dict"]
    load_adapters(args, pipe, relighting_lora_state_dict)

    df = pd.read_csv(args.test_csv)
    if args.sample_num > 0:
        df = df.sample(n=args.sample_num, random_state=args.seed).reset_index(drop=True)
    elif not (getattr(args, "val_subset", 0) or getattr(args, "val_ids_file", None)):
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    # Validation subset: same construction as training (first N unique clip_id in CSV order, or ids from val_ids_file)
    df = build_val_subset_df(df, val_subset=getattr(args, "val_subset", 0), val_ids_file=getattr(args, "val_ids_file", None))
    if (getattr(args, "val_subset", 0) > 0 or getattr(args, "val_ids_file", None)) and rank == 0:
        print(f"[INFO] Using validation subset: {len(df)} samples (same logic as training val)")
    print(f"[Rank {rank}] [INFO] Total test samples: {len(df)}")

    # Align num_frames with pipeline (same as runner): (num_frames - 1) % 4 == 0
    num_frames = max(1, args.num_frames)
    if (num_frames - 1) % 4 != 0:
        num_frames = (num_frames + 2) // 4 * 4 + 1
    target_len = max(1, num_frames - 4)  # conditioning length pipeline expects

    def _to_len(lst, length):
        if lst is None or not isinstance(lst, (list, tuple)):
            return lst
        if len(lst) >= length:
            return list(lst)[:length]
        out = list(lst)
        last = out[-1] if out else None
        while len(out) < length and last is not None:
            out.append(last)
        return out

    res_map = {
        "openhuman": (720, 1280), # height, width
        "short-drama": (1280, 720),
        "visko": (1280, 720)
    }

    has_csv_ref_image = args.input_ref_key in df.columns

    if world_size > 1:
        indices = np.array_split(np.arange(len(df)), world_size)
        df_shard = df.iloc[indices[rank]]
        print(f"[Rank {rank}] Processing {len(df_shard)} samples")
    else:
        df_shard = df
        print(f"[Rank {rank}] Processing {len(df_shard)} samples")

    gt_for_metrics = []
    pred_for_metrics = []
    clip_ids_for_metrics = []

    def _serialize_clip_id(cid):
        if isinstance(cid, (list, tuple)):
            return "_".join(str(x) for x in cid)
        return str(cid)

    desc_text = f"Rank {rank} Infer"
    for _, row in tqdm(df_shard.iterrows(), total=len(df_shard), desc=desc_text, position=rank):
        clip_id = row['clip_id']
        source = row['source']
        prompt = row['prompt']
        video_path = row['video']
        
        mask_type = row.get('mask_type')
        mask_suffix = f"_{mask_type}" if pd.notna(mask_type) and str(mask_type).strip() != "" else ""
        
        try:
            clip_dir = os.path.join(args.output_dir, str(clip_id))
            os.makedirs(clip_dir, exist_ok=True)
            
            dst_video_path = os.path.join(clip_dir, os.path.basename(str(video_path)))
            if not os.path.exists(dst_video_path):
                shutil.copy2(str(video_path), dst_video_path)

            ref_image_path = get_ref_image_path(args, row, has_csv_ref_image, args.input_ref_key, rank=rank)
            if os.path.exists(str(ref_image_path)):
                dst_ref_path = os.path.join(clip_dir, os.path.basename(str(ref_image_path)))
                if not os.path.exists(dst_ref_path):
                    shutil.copy2(str(ref_image_path), dst_ref_path)

            output_filename = f"{args.prefix_video_name}_ID-{ref_name}{mask_suffix}.mp4"
            output_path = os.path.join(clip_dir, output_filename)
            
            if os.path.exists(output_path):
                continue

            video_width, video_height = get_video_size(video_path)
            
            ref_image = Image.open(ref_image_path).convert("RGB")
            input_image = ref_image.resize((video_width, video_height), Image.BICUBIC)

            pose_video = _to_len(VideoData(row['animate_pose_video']).raw_data(), target_len)
            face_video = _to_len(VideoData(row['animate_face_video']).raw_data(), target_len)
            inpaint_video = _to_len(VideoData(row['animate_inpaint_video']).raw_data(), target_len)
            mask_video = _to_len(VideoData(row['animate_mask_video']).raw_data(), target_len)

            if source not in res_map:
                raise ValueError(f"Unsupported source: {source}")
            h, w = res_map[source]

            with torch.no_grad():
                pipe_kwargs = dict(
                    prompt=prompt,
                    seed=args.seed,
                    tiled=False,
                    input_image=input_image,
                    animate_pose_video=pose_video,
                    animate_face_video=face_video,
                    animate_inpaint_video=inpaint_video,
                    animate_mask_video=mask_video,
                    num_frames=num_frames,
                    height=h,
                    width=w,
                    num_inference_steps=args.steps,
                    cfg_scale=args.cfg,
                )
                if args.mode in {"arcface", "arcface_dit_lora"}:
                    pipe_kwargs["animate_ref_identity_image"] = ref_image
                video = pipe(**pipe_kwargs)

            # 图像尺寸对齐：生成视频可能为 res_map 的 (h,w)，GT 为 (video_width, video_height)；统一 resize 到 GT 尺寸后再保存和算指标
            video_out = [f.resize((video_width, video_height), Image.BICUBIC) for f in video]
            save_video(video_out, output_path, fps=15, quality=5)
            print(f"[Rank {rank}] [INFO] Video saved to {output_path}")

            if args.compute_metrics:
                cap_per_rank = max(1, args.metrics_max_videos // max(1, world_size))
                if len(gt_for_metrics) < cap_per_rank:
                    gt_loader = VideoData(video_path)
                    gt_frames = gt_loader.raw_data()[:target_len]
                    if hasattr(gt_loader.data, "reader") and gt_loader.data.reader is not None:
                        gt_loader.data.reader.close()
                    if len(gt_frames) >= 1 and len(video_out) >= 1:
                        gt_for_metrics.append(gt_frames)
                        pred_for_metrics.append(video_out)
                        clip_ids_for_metrics.append(_serialize_clip_id(clip_id))
            
        except Exception as e:
            print(f"\n[Rank {rank}] [ERROR] Skipping {clip_id} due to error: {e}")
            continue

    if args.compute_metrics:
        if len(gt_for_metrics) > 0:
            real_tensors = [video_to_tensor(g).cpu() for g in gt_for_metrics]
            pred_tensors = [video_to_tensor(p).cpu() for p in pred_for_metrics]
            real_all, pred_all, ids_all = gather_video_pairs_to_rank0(
                rank, world_size, real_tensors, pred_tensors, id_list=clip_ids_for_metrics
            )
            if rank == 0:
                if real_all and pred_all and len(real_all) > 0 and len(pred_all) > 0:
                    detailed = compute_video_metrics_detailed(
                        real_all,
                        pred_all,
                        device=device,
                        data_range=1.0,
                        compute_fvd=True,
                        fvd_batch_size=2,
                        max_fvd_samples=args.metrics_max_videos,
                    )
                    # per_video: list of {clip_id, psnr, ssim}; summary: num_valid, mean_psnr, mean_ssim, fvd
                    ids_all = ids_all if ids_all is not None else [f"video_{i}" for i in range(len(detailed["per_video"]))]
                    per_video = []
                    for i, pv in enumerate(detailed["per_video"]):
                        entry = {"clip_id": ids_all[i] if i < len(ids_all) else f"video_{i}"}
                        entry.update(pv)
                        per_video.append(entry)
                    output_metrics = {
                        "per_video": per_video,
                        "summary": detailed["summary"],
                    }
                    metrics_path = os.path.join(args.output_dir, "metrics.json")
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        import json
                        json.dump(output_metrics, f, indent=2, ensure_ascii=False)
                    s = detailed["summary"]
                    print(f"[Rank 0] [METRICS] num_valid={s['num_videos_with_valid_metrics']} "
                          f"mean_PSNR={s['mean_psnr']:.4f} mean_SSIM={s['mean_ssim']:.4f} FVD={s['fvd']:.4f} -> {metrics_path}")
                else:
                    print(f"[Rank 0] [WARN] No valid video pairs for metrics (real_all: {len(real_all) if real_all else 0}, pred_all: {len(pred_all) if pred_all else 0})")
        else:
            if rank == 0:
                print(f"[Rank 0] [WARN] --compute_metrics enabled but no GT videos collected (gt_for_metrics is empty)")

    if world_size > 1:
        dist.destroy_process_group()

    print(f"\n[Rank {rank}] [FINISH] Inference results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
