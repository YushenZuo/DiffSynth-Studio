import os, torch, math, functools
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    ShardingStrategy, 
    MixedPrecision, 
    StateDictType,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from accelerate import Accelerator, FullyShardedDataParallelPlugin

from diffusers.optimization import get_scheduler
from safetensors.torch import load_file as load_safetensors

from ..models.wan_video_dit import DiTBlock
from .logger import ModelLogger
from .training_module import DiffusionTrainingModule


def get_fsdp_plugin(fsdp_target_models):
    if fsdp_target_models == "dit":
        wan_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={DiTBlock},)
    
    wan_mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    sharded_model_config = ShardedStateDictConfig(offload_to_cpu=True)
    sharded_optim_config = ShardedOptimStateDictConfig(offload_to_cpu=True)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wan_auto_wrap_policy,
        mixed_precision_policy=wan_mixed_precision,
        state_dict_type=StateDictType.SHARDED_STATE_DICT, # 必须使用分片模式
        state_dict_config=sharded_model_config,          # 传入原生配置对象
        optim_state_dict_config=sharded_optim_config,    # 传入原生优化器配置
        use_orig_params=True,                            # 必须为 True 以支持梯度检查点
        sync_module_states=True,                         # 多机启动同步
        limit_all_gathers=True,                          # 对应定义中的参数名，限制通信压力
    )

    return fsdp_plugin


def launch_fsdp_training_task(
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

    # Advanced Training Strtegy, FSDP
    if accelerator.is_main_process:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        is_fsdp = isinstance(model, FSDP)
        print(f"Model is FSDP wrapped: {is_fsdp}")
    use_fsdp_v1 = args.advanced_parallel_strategy in ["fsdp", "fsdp_usp"]
    use_fsdp_v2 = args.advanced_parallel_strategy == "fsdp_v2"
    
    model.train()
    
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
                optimizer.zero_grad(set_to_none=True)
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
                
                model_logger.on_step_end(accelerator, model, save_steps, use_fsdp_v1=use_fsdp_v1)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id, use_fsdp_v1=use_fsdp_v1)
    model_logger.on_training_end(accelerator, model, save_steps, use_fsdp_v1=use_fsdp_v1)
    accelerator.end_training()