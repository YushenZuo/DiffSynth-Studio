import argparse

def get_wan_base_parser():
    parser = argparse.ArgumentParser(description="Wan Video Diffusion Training Script (DiffSynth)", add_help=False)

    parser.add_argument("--dataset_base_path", type=str, default=None, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the dataset metadata file.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--data_file_keys", type=str, default="video,animate_pose_video,animate_face_video,animate_inpaint_video,animate_mask_video,animate_ref_identity_image", help="Data file keys in metadata, comma-separated.")

    parser.add_argument("--height", type=int, default=None, help="Target height. Leave empty for dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Target width. Leave empty for dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=832*480, help="Maximum pixels per frame for dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video (sampled from prefix).")

    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models (JSON format).")
    parser.add_argument("--model_id_with_origin_paths", type=str, default="Wan-AI/Wan2.2-Animate-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-Animate-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-Animate-14B:Wan2.1_VAE.pth,Wan-AI/Wan2.2-Animate-14B:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", help="Model ID with origin paths, e.g. Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors")
    parser.add_argument("--extra_inputs", type=str, default="input_image,animate_pose_video,animate_face_video,animate_inpaint_video,animate_mask_video,animate_ref_identity_image", help="Additional model inputs, comma-separated.")
    parser.add_argument("--fp8_models", type=str, default=None, help="Models using FP8 precision, comma-separated.")
    parser.add_argument("--offload_models", type=str, default=None, help="Models to offload to CPU (for split training).")

    parser.add_argument("--task", type=str, default="sft:train", help="Training task type (e.g., sft, direct_distill).")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g. dit, vae, text_encoder.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--find_unused_parameters", action="store_true", help="Enable find_unused_parameters in DDP.")

    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=True, action="store_true", help="Offload gradient checkpointing to CPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")

    parser.add_argument("--lora_base_model", type=str, default=None, help="Base model to apply LoRA.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Target modules for LoRA.")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to LoRA checkpoint to resume from.")
    parser.add_argument("--preset_lora_path", type=str, default=None, help="Path to preset LoRA checkpoint (fused to base model).")
    parser.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")

    parser.add_argument("--output_path", type=str, default="./models/train/Wan2.2-Animate-14B_full", help="Output directory for checkpoints.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default=None, help="Prefix to remove when saving checkpoints.")
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint saving interval (steps).")

    parser.add_argument("--tokenizer_path", type=str, default="./models/Wan-AI/Wan2.2-Animate-14B/google/umt5-xxl", help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to audio processor (Wan2.2-S2V).")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary for mixed Wan models.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary for mixed Wan models.")
    parser.add_argument("--initialize_model_on_cpu", action="store_true", help="Initialize models on CPU first.")

    parser.add_argument("--max_grad_norm", type=float, default=1.0,)
    parser.add_argument("--val_num_videos", type=int, default=16, help="Number of videos to use for periodic validation.")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Batch size for validation inference.")
    parser.add_argument("--val_inference_steps", type=int, default=20, help="Inference steps during validation (align with inference.py --steps).")
    parser.add_argument("--val_cfg_scale", type=float, default=1.0, help="CFG scale during validation (align with inference.py --cfg).")
    parser.add_argument("--val_seed", type=int, default=0, help="Random seed for validation inference (align with inference.py --seed).")
    parser.add_argument("--wandb_entity", type=str, default="mingyang__", help="WandB team/organization name")
    parser.add_argument("--wandb_project", type=str, default="Visko-VideoGen")
    parser.add_argument("--wandb_run_name", type=str, default="Training-FSDP")

    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to LoRA / trainable-only safetensors to resume from")
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help='The scheduler type to use.',)

    return parser
