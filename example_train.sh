# Example Usage (train + periodic validation: PSNR/SSIM/FVD every save_steps, videos under output_path/val/step-*)
# 验证训练部分时：保留 --val_num_videos 和 --save_steps，跑满 500 step 后会在 output_path/val/step-500/ 存 GT/生成视频，并在 wandb 打 val/psnr, val/ssim, val/fvd
CUDA_VISIBLE_DEVICES=4,5 /mnt/beegfs/yushen/miniconda3/envs/vgr/bin/python -m accelerate.commands.launch \
  --num_processes 2 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29501 \
  train.py \
    --dataset_base_path /mnt/beegfs/yushen/research/DiffSynth-Studio/data/processed_feature/metadata_replace_20260103_191744 \
    --dataset_metadata_path /mnt/beegfs/yushen/research/DiffSynth-Studio/data/training_metadata/data-20260103_19/metadata_replace_20260103_191744.csv \
    --data_file_keys "video,animate_pose_video,animate_face_video,animate_inpaint_video,animate_mask_video,animate_ref_identity_image" \
    --num_frames 49 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-Animate-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-Animate-14B:Wan2.1_VAE.pth,Wan-AI/Wan2.2-Animate-14B:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --tokenizer_path /mnt/beegfs/yushen/research/DiffSynth-Studio/models/Wan-AI/Wan2.2-Animate-14B/google/umt5-xxl \
    --learning_rate 1e-4 \
    --lr_scheduler constant \
    --num_epochs 1 \
    --output_path "/mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/debug_metric_9" \
    --trainable_models "animate_adapter" \
    --extra_inputs "input_image,animate_pose_video,animate_face_video,animate_inpaint_video,animate_mask_video,animate_ref_identity_image" \
    --dataset_num_workers 8 \
    --gradient_accumulation_steps 1 \
    --find_unused_parameters \
    --task sft \
    --save_steps 10 \
    --val_num_videos 4 \
    --val_batch_size 1 \
    --val_inference_steps 20 \
    --val_cfg_scale 1.0 \
    --use_ref_as_input \
    --ref_image_key animate_ref_identity_image \
    --wandb_entity mingyang__ \
    --wandb_project Visko-VideoGen-Yushen \
    --wandb_run_name Wan-Animate-Continue-Train-Metric-9
    

CUDA_VISIBLE_DEVICES=0,1 /mnt/beegfs/yushen/miniconda3/envs/vgr/bin/python -m accelerate.commands.launch \
  --num_processes 2 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29501 \
  train_arcface.py \
      --dataset_base_path /mnt/beegfs/yushen/research/DiffSynth-Studio/data/processed_feature/metadata_replace_20260103_191744 \
      --dataset_metadata_path /mnt/beegfs/yushen/research/DiffSynth-Studio/data/training_metadata/data-20260103_19/metadata_replace_20260103_191744.csv \
      --data_file_keys "animate_ref_identity_image" \
      --num_frames 49 \
      --dataset_repeat 1 \
      --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-Animate-14B:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
      --tokenizer_path /mnt/beegfs/yushen/research/DiffSynth-Studio/models/Wan-AI/Wan2.2-Animate-14B/google/umt5-xxl \
      --learning_rate 1e-4 \
      --lr_scheduler constant \
      --num_epochs 3 \
      --output_path "/mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/debug" \
      --trainable_models "face_encoder,animate_arcface_adapter" \
      --extra_inputs "animate_ref_identity_image" \
      --dataset_num_workers 4 \
      --gradient_accumulation_steps 1 \
      --find_unused_parameters \
      --task sft:train2 \
      --save_steps 250 \
      --lora_base_model "dit" \
      --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
      --lora_rank 128 \
      --load_metadata_and_cache \
      --face_model_path /mnt/beegfs/yushen/research/DiffSynth-Studio \
      --use_ref_as_input \
      --ref_image_key animate_ref_identity_image \
      --wandb_entity mingyang__ \
      --wandb_project Visko-VideoGen-Yushen \
      --wandb_run_name Wan-Animate-Arcface-Train-Yushen