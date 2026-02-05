# ========== Test metrics on validation subset with pretrained model (same construction as training val) ==========
# Uses first N unique clip_id in CSV order, or --val_ids_file path/to/val/val_ids.json from training.
# For pretrained model (no checkpoint): use --mode base or --mode base_wo_relighting
CUDA_VISIBLE_DEVICES=6 python inference.py \
  --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/training_metadata/data-20260103_19/metadata_replace_20260103_191744.csv \
  --output_dir ./results/pretrained_val_metrics \
  --mode base --num_frames 49 --steps 20 \
  --val_subset 4 --compute_metrics --metrics_max_videos 4 \
  --input_ref_key animate_ref_identity_image

# ========== Inference + PSNR/SSIM/FVD (single GPU) ==========
# CUDA_VISIBLE_DEVICES=0 python inference.py \
#   --test_csv /path/to/test.csv \
#   --output_dir ./results/eval \
#   --checkpoint /path/to/step-500.safetensors \
#   --mode ft_dit_lora --num_frames 81 --steps 20 \
#   --compute_metrics --metrics_max_videos 200

# ========== Inference + metrics with multi-GPU (gather to rank 0, then compute) ==========
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#   --test_csv /path/to/test.csv \
#   --output_dir ./results/eval \
#   --checkpoint /path/to/step-500.safetensors \
#   --mode ft_dit_lora --num_frames 81 --steps 20 \
#   --compute_metrics --metrics_max_videos 200

# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/training_metadata_for_reproduce/data-20260115_23/metadata_val_20260115_232041.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce/metadata_val_20260115_232041 \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce/v0.0_mask_loss_dropout/20260119_001928-DiT_LoRA128-Data_20260115-Ref_Key_Drop_Aug/step-650.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 30 \
#   --prefix_video_name Wan-Animate_Ref-Segment_step-650_mask_loss_drop_aug_v3_denoise_30 \
#   --input_ref_key animate_ref_identity_image

# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/training_metadata_for_reproduce/data-20260115_23/metadata_val_20260115_232041.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce/metadata_val_20260115_232041 \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce/v0.0_mask_loss_dropout/20260118_112403-DiT_LoRA128-Data_20260115-Ref_Key_Drop_Aug/step-750.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Ref-Segment_step-750_mask_loss_drop_aug_v2_denoise_30 \
#   --input_ref_key animate_ref_identity_image

# CUDA_VISIBLE_DEVICES=4 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce/metadata_visko_test_20260108_232611_map_ref_image \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce/v0.0_mask_loss_dropout/20260119_210309-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-1000.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_Ref-Segment_step-1000_mask_loss_drop_aug_v3 \
#   --input_ref_key animate_ref_identity_image

# CUDA_VISIBLE_DEVICES=7 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce/metadata_visko_test_20260108_232611_map_ref_image \
#   --mode base \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name baseline \
#   --input_ref_key animate_ref_identity_image


# # v1
# CUDA_VISIBLE_DEVICES=1 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce2/metadata_visko_test_20260108_232611_map_ref_image \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week2/v0.0_mask_loss_dropout/20260121_064638-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-750.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_step-750_mask_loss_drop_aug_v3_reduce_light \
#   --input_ref_key animate_ref_identity_image


# CUDA_VISIBLE_DEVICES=2 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce2/metadata_visko_test_20260108_232611_map_ref_image \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week2/v0.0_mask_loss_dropout/20260121_064638-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-2000.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_step-2000_mask_loss_drop_aug_v3_reduce_light \
#   --input_ref_key animate_ref_identity_image


# CUDA_VISIBLE_DEVICES=4 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce2/metadata_visko_test_20260108_232611_map_ref_image \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week2/v0.0_mask_loss_dropout/20260121_064638-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-1500.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_step-1500_mask_loss_drop_aug_v3_reduce_light \
#   --input_ref_key animate_ref_identity_image


# # V2
# CUDA_VISIBLE_DEVICES=1 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce2/metadata_visko_test_20260108_232611_map_ref_image \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week2/v0.0_mask_loss_dropout/20260121_065044-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-2000.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_step-2000_mask_loss_drop_aug_v2_reduce_light \
#   --input_ref_key animate_ref_identity_image

# CUDA_VISIBLE_DEVICES=2 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce2/metadata_visko_test_20260108_232611_map_ref_image \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week2/v0.0_mask_loss_dropout/20260121_065044-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-1000.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_step-1000_mask_loss_drop_aug_v2_reduce_light \
#   --input_ref_key animate_ref_identity_image

# CUDA_VISIBLE_DEVICES=4 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce2/metadata_visko_test_20260108_232611_map_ref_image \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week2/v0.0_mask_loss_dropout/20260121_065044-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-1500.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_step-1500_mask_loss_drop_aug_v2_reduce_light \
#   --input_ref_key animate_ref_identity_image

# CUDA_VISIBLE_DEVICES=6 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/training_metadata_for_reproduce/data-20260115_23/metadata_val_20260115_232041.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce/metadata_val_20260115_232041 \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week2/v0.0_mask_loss_dropout/20260121_065044-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-1750.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_step-1750_mask_loss_drop_aug_v2_reduce_light \
#   --input_ref_key animate_ref_identity_image &

# CUDA_VISIBLE_DEVICES=7 python inference.py \
#   --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/testing_metadata/metadata_visko_test_20260108_232611_map_ref_image.csv \
#   --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/for_reproduce2/metadata_visko_test_20260108_232611_map_ref_image \
#   --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week2/v0.0_mask_loss_dropout/20260121_065044-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug/step-3000.safetensors \
#   --mode ft_dit_lora \
#   --num_frames 81 \
#   --steps 20 \
#   --prefix_video_name Wan-Animate_Data-20260103_step-3000_mask_loss_drop_aug_v2_reduce_light \
#   --input_ref_key animate_ref_identity_image &

# wait



# inference reproduce for week3

# bash inference.sh

### cross pair
steps=(50 500 750 100 250 1000 1250 2000)
for step in "${steps[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 torchrun --nproc_per_node=4 inference.py \
    --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/validation/testing_metadata/20260201/curated_inpair_20260201_samples_80_ratio_3_1.csv \
    --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/reproduce_week3/curated_inpair_20260201_samples_80_ratio_3_1 \
    --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week3/v0.0_dropout/20260201_112434-DiT_LoRA128-Data_20260103_19_25k-Ref_First/step-${step}.safetensors \
    --mode ft_dit_lora \
    --num_frames 81 \
    --steps 20 \
    --prefix_video_name 20260201_112434_step-${step} \
    --input_ref_key animate_ref_identity_image
done


steps=(50 500 750 100 250 1000 1250 2000)
for step in "${steps[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 torchrun --nproc_per_node=4 inference.py \
    --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/validation/testing_metadata/20260201/curated_inpair_20260201_samples_80_ratio_3_1.csv \
    --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/reproduce_week3/curated_inpair_20260201_samples_80_ratio_3_1 \
    --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week3/v0.0_dropout/20260201_111456-DiT_LoRA128-Data_20260103_19_25k-Ref_First_Drop_Aug_wo_light/step-${step}.safetensors \
    --mode ft_dit_lora \
    --num_frames 81 \
    --steps 20 \
    --prefix_video_name 20260201_111456_step-${step} \
    --input_ref_key animate_ref_identity_image
done


steps=(50 500 750 100 250 1000 1250 2000)
for step in "${steps[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 torchrun --nproc_per_node=4 inference.py \
    --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/validation/testing_metadata/20260201/cross_pair_visko.csv \
    --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/reproduce_week3/cross_pair_visko \
    --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week3/v0.0_dropout/20260201_112434-DiT_LoRA128-Data_20260103_19_25k-Ref_First/step-${step}.safetensors \
    --mode ft_dit_lora \
    --num_frames 81 \
    --steps 20 \
    --prefix_video_name 20260201_112434_step-${step} \
    --input_ref_key inference_ref_image
done


steps=(50 500 750 100 250 1000 1250 2000)
for step in "${steps[@]}"; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 torchrun --nproc_per_node=4 inference.py \
    --test_csv /mnt/beegfs/yushen/research/DiffSynth-Studio/data/validation/testing_metadata/20260201/cross_pair_visko.csv \
    --output_dir /mnt/beegfs/yushen/research/DiffSynth-Studio/results/reproduce_week3/cross_pair_visko \
    --checkpoint /mnt/beegfs/yushen/research/DiffSynth-Studio/models/train/reproduce_week3/v0.0_dropout/20260201_111456-DiT_LoRA128-Data_20260103_19_25k-Ref_First_Drop_Aug_wo_light/step-${step}.safetensors \
    --mode ft_dit_lora \
    --num_frames 81 \
    --steps 20 \
    --prefix_video_name 20260201_111456_step-${step} \
    --input_ref_key inference_ref_image
done

