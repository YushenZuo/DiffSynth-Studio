import os, math, json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from safetensors.torch import load_file as load_safetensors
from diffusers.optimization import get_scheduler

from diffsynth.utils.data import save_video
from diffsynth.metrics.video_metrics import video_to_tensor, compute_video_metrics
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


# def launch_training_task(
#     accelerator: Accelerator,
#     dataset: torch.utils.data.Dataset,
#     model: DiffusionTrainingModule,
#     model_logger: ModelLogger,
#     learning_rate: float = 1e-5,
#     weight_decay: float = 1e-2,
#     num_workers: int = 1,
#     save_steps: int = None,
#     num_epochs: int = 1,
#     args = None,
# ):
#     if args is not None:
#         learning_rate = args.learning_rate
#         weight_decay = args.weight_decay
#         num_workers = args.dataset_num_workers
#         save_steps = args.save_steps
#         num_epochs = args.num_epochs
    
#     optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
#     dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
#     model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
#     for epoch_id in range(num_epochs):
#         for data in tqdm(dataloader):
#             with accelerator.accumulate(model):
#                 optimizer.zero_grad()
#                 if dataset.load_from_cache:
#                     loss = model({}, inputs=data)
#                 else:
#                     loss = model(data)
#                 accelerator.backward(loss)
#                 optimizer.step()
#                 model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
#                 scheduler.step()
#         if save_steps is None:
#             model_logger.on_epoch_end(accelerator, model, epoch_id)
#     model_logger.on_training_end(accelerator, model, save_steps)

def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    val_dataset: torch.utils.data.Dataset | None = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    if accelerator.is_main_process:
        init_kwargs = {"wandb": {"entity": args.wandb_entity, "name": args.wandb_run_name}}
        accelerator.init_trackers(args.wandb_project, config=vars(args) if args else {}, init_kwargs=init_kwargs)
    
    # optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    if args.lr_scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    else:
        steps_per_epoch = math.ceil(len(dataloader) / (args.gradient_accumulation_steps * accelerator.num_processes))
        if getattr(args, "max_train_steps", None) is None or args.max_train_steps <= 0:
            args.max_train_steps = num_epochs * steps_per_epoch
        if not hasattr(args, "lr_warmup_steps") or args.lr_warmup_steps is None or args.lr_warmup_steps == 0:
            args.lr_warmup_steps = int(args.max_train_steps * 0.05)
        scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    model_logger.attach_training_components(optimizer, scheduler)

    model.train()

    # Step 0 validation: always run before loading any checkpoint, so step 0 = pretrained model metrics
    if val_dataset is not None and getattr(args, "val_num_videos", 0) > 0:
        evaluate_on_validation(accelerator, model, val_dataset, args, global_step=0)

    if args.resume_checkpoint is not None:
        accelerator.print(f"[INFO] Loading trainable checkpoint from {args.resume_checkpoint}")
        if args.resume_checkpoint.endswith(".safetensors"):
            checkpoint = {"model": load_safetensors(args.resume_checkpoint, device="cpu")}
        else:
            checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model_checkpoint = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        accelerator.unwrap_model(model).load_state_dict(model_checkpoint, strict=False)
        if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and isinstance(checkpoint, dict) and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if isinstance(checkpoint, dict):
            model_logger.num_steps = checkpoint.get("step", model_logger.num_steps)
            model_logger.current_epoch = checkpoint.get("epoch", model_logger.current_epoch)

    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id}", disable=not accelerator.is_main_process)
        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)

                grad_norm = 0.0
                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        params = [p for p in model.parameters() if p.grad is not None]
                        if params:
                            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2).item()
                
                optimizer.step()
                scheduler.step()

                if accelerator.sync_gradients:
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    param_norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in trainable_params]), 2).item()
                    log_data = {
                        "train/loss": accelerator.gather(loss.detach()).mean().item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm, "train/param_norm": param_norm
                    }
                    if getattr(args, "lora_base_model", None) is not None:
                        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
                        if lora_params:
                            lora_norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in lora_params]), 2).item()
                            log_data["train/lora_param_norm"] = lora_norm
                    accelerator.log(log_data, step=model_logger.num_steps)
                    pbar.set_postfix({"loss": f"{log_data['train/loss']:.4f}", "gnorm": f"{grad_norm:.2f}"})
                
                # Step-level checkpointing
                model_logger.on_step_end(accelerator, model, save_steps)

                # Periodic validation on a held-out subset
                if (
                    val_dataset is not None
                    and save_steps is not None
                    and model_logger.num_steps % save_steps == 0
                ):
                    evaluate_on_validation(accelerator, model, val_dataset, args, global_step=model_logger.num_steps)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    accelerator.end_training()


def evaluate_on_validation(
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    val_dataset: torch.utils.data.Dataset,
    args,
    global_step: int,
):
    """
    Run validation on a small subset of the training data and log PSNR / SSIM / FVD.
    """
    if val_dataset is None:
        return

    if getattr(args, "val_num_videos", 0) <= 0:
        return

    # Sync: only main process runs validation; others wait here to avoid NCCL timeout (main in validation, others in backward).
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    from torch.utils.data import DataLoader

    unwrapped_model = accelerator.unwrap_model(model)
    pipe = getattr(unwrapped_model, "pipe", None)
    if pipe is None:
        accelerator.print("[WARN] Validation skipped: model has no `pipe` attribute.")
        return

    # Validation set: use stable ids (clip_id / path) when available, else indices
    if isinstance(val_dataset, torch.utils.data.Subset):
        base_dataset = val_dataset.dataset
        val_indices = list(val_dataset.indices)
        val_ids = None
        if hasattr(base_dataset, "get_stable_id"):
            val_ids = [base_dataset.get_stable_id(i) for i in val_indices]
    else:
        base_dataset = val_dataset
        val_indices = list(range(len(val_dataset)))
        val_ids = [base_dataset.get_stable_id(i) for i in val_indices] if hasattr(base_dataset, "get_stable_id") else None
    val_seed = getattr(args, "val_seed", 0)
    accelerator.print(f"[VAL] Evaluating model at step {global_step} ...")
    accelerator.print(f"[VAL] Validation samples (ids): {val_ids if val_ids is not None else val_indices}")
    if getattr(args, "output_path", None) is not None:
        os.makedirs(os.path.join(args.output_path, "val"), exist_ok=True)
        ids_path = os.path.join(args.output_path, "val", "val_ids.json")
        to_save = {"val_ids": [list(s) if isinstance(s, tuple) else s for s in val_ids]} if val_ids is not None else {"val_indices": val_indices}
        to_save["val_num_videos"] = len(val_indices)
        with open(ids_path, "w") as f:
            json.dump(to_save, f, indent=2)
        accelerator.print(f"[VAL] Saved validation to {ids_path}")

    num_workers = getattr(args, "dataset_num_workers", 0)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
    )

    real_videos = []
    fake_videos = []

    device = pipe.device if isinstance(pipe.device, torch.device) else torch.device(str(pipe.device))
    val_video_root = None
    if hasattr(args, "output_path") and args.output_path is not None:
        val_video_root = os.path.join(args.output_path, "val", f"step-{global_step}")
        os.makedirs(val_video_root, exist_ok=True)

    unwrapped_model.eval()
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            if idx >= args.val_num_videos:
                break

            if "video" not in data or "prompt" not in data:
                continue

            gt_video = data["video"]  # list of PIL.Image

            # Reuse the same input construction logic as in training
            inputs_shared, inputs_posi, inputs_nega = unwrapped_model.get_pipeline_inputs(data)

            # Use fixed num_frames from args so pipeline latent and y stay aligned (pipeline requires num_frames % 4 == 1).
            # Align with inference.py: target_len = args.num_frames - 4 for conditioning videos, pipe gets num_frames=args.num_frames.
            val_num_frames = max(1, getattr(args, "num_frames", 81))
            if (val_num_frames - 1) % 4 != 0:
                val_num_frames = (val_num_frames + 2) // 4 * 4 + 1
            target_len = max(1, val_num_frames - 4)  # same as inference.py: conditioning videos truncated to num_frames - 4
            height = inputs_shared["height"]
            width = inputs_shared["width"]

            def _to_len(lst, target_len):
                if lst is None or not isinstance(lst, (list, tuple)):
                    return lst
                if len(lst) >= target_len:
                    return list(lst)[:target_len]
                out = list(lst)
                last = out[-1] if out else None
                while len(out) < target_len and last is not None:
                    out.append(last)
                return out

            for key in ("animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"):
                if inputs_shared.get(key) is not None:
                    inputs_shared[key] = _to_len(inputs_shared[key], target_len)
            num_frames = val_num_frames

            # Align with inference.py: same parameters as inference.py pipe call
            val_steps = getattr(args, "val_inference_steps", 20)
            val_cfg = getattr(args, "val_cfg_scale", 1.0)
            val_seed = getattr(args, "val_seed", 0)
            pipe_kwargs = dict(
                prompt=inputs_posi["prompt"],
                seed=val_seed,
                tiled=False,
                input_image=inputs_shared.get("input_image"),
                animate_pose_video=inputs_shared.get("animate_pose_video"),
                animate_face_video=inputs_shared.get("animate_face_video"),
                animate_inpaint_video=inputs_shared.get("animate_inpaint_video"),
                animate_mask_video=inputs_shared.get("animate_mask_video"),
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=val_steps,
                cfg_scale=val_cfg,
            )
            # Optional: animate_ref_identity_image (if available, e.g. for arcface mode)
            if inputs_shared.get("reference_image") is not None:
                pipe_kwargs["animate_ref_identity_image"] = inputs_shared["reference_image"]
            generated_video = pipe(**pipe_kwargs)

            # Align number of frames between GT and prediction
            aligned_frames = min(len(gt_video), len(generated_video))
            if aligned_frames == 0:
                continue
            gt_video_aligned = gt_video[:aligned_frames]
            pred_video_aligned = generated_video[:aligned_frames]

            # 保存当前样本的视频（GT 和 生成）；优先用 clip_id 作为文件名，便于对照
            if val_video_root is not None:
                raw_id = data.get("clip_id", data.get("id", None))
                if raw_id is not None:
                    # 元组/列表（如 (stem, parent)）或路径转为安全文件名
                    base_name = "_".join(str(x) for x in raw_id) if isinstance(raw_id, (list, tuple)) else str(raw_id)
                    base_name = base_name.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
                else:
                    base_name = f"sample-{idx:04d}"
                gt_path = os.path.join(val_video_root, f"{base_name}_gt.mp4")
                pred_path = os.path.join(val_video_root, f"{base_name}_pred.mp4")
                try:
                    save_video(gt_video_aligned, gt_path, fps=15, quality=5)
                    save_video(pred_video_aligned, pred_path, fps=15, quality=5)
                except Exception as e:  # pragma: no cover - best-effort saving
                    accelerator.print(f"[WARN] Failed to save validation videos for {base_name}: {e}")

            gt_tensor = video_to_tensor(gt_video_aligned)
            pred_tensor = video_to_tensor(pred_video_aligned)
            real_videos.append(gt_tensor)
            fake_videos.append(pred_tensor)

    unwrapped_model.train()

    if not real_videos or not fake_videos:
        accelerator.print("[WARN] Validation produced no usable samples.")
        accelerator.wait_for_everyone()
        return

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
    psnr, ssim, fvd = metrics_dict["psnr"], metrics_dict["ssim"], metrics_dict["fvd"]
    accelerator.print(f"[VAL] step {global_step} | PSNR={psnr:.4f} SSIM={ssim:.4f} FVD={fvd:.4f}")

    # Load / update / save best-so-far (PSNR/SSIM: higher better, FVD: lower better)
    best_path = None
    if getattr(args, "output_path", None) is not None:
        os.makedirs(os.path.join(args.output_path, "val"), exist_ok=True)
        best_path = os.path.join(args.output_path, "val", "best_metrics.json")
    best = {}
    if best_path and os.path.isfile(best_path):
        try:
            with open(best_path, "r") as f:
                best = json.load(f)
        except Exception:
            best = {}
    best.setdefault("best_psnr", {"value": float("-inf"), "step": -1})
    best.setdefault("best_ssim", {"value": float("-inf"), "step": -1})
    best.setdefault("best_fvd", {"value": float("inf"), "step": -1})

    def _update_best(key, current, step, higher_better):
        if current != current:  # nan
            return
        prev = best[key]["value"]
        if higher_better and current > prev:
            best[key] = {"value": current, "step": step}
        elif not higher_better and current < prev:
            best[key] = {"value": current, "step": step}

    _update_best("best_psnr", psnr, global_step, higher_better=True)
    _update_best("best_ssim", ssim, global_step, higher_better=True)
    _update_best("best_fvd", fvd, global_step, higher_better=False)

    if best_path:
        try:
            with open(best_path, "w") as f:
                json.dump(best, f, indent=2)
        except Exception:
            pass

    bp, bs, bf = best["best_psnr"], best["best_ssim"], best["best_fvd"]
    accelerator.print(
        f"[VAL] Best so far: PSNR {bp['value']:.4f} (step {bp['step']}), "
        f"SSIM {bs['value']:.4f} (step {bs['step']}), FVD {bf['value']:.4f} (step {bf['step']})"
    )

    # Restore pipeline to training mode (validation pipe() sets scheduler.training=False, so next forward would miss input_latents)
    unwrapped = accelerator.unwrap_model(model)
    pipe = getattr(unwrapped, "pipe", None)
    if pipe is not None and getattr(pipe, "scheduler", None) is not None:
        pipe.scheduler.set_timesteps(1000, training=True)

    accelerator.wait_for_everyone()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
