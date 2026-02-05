import torch
import torch.nn.functional as F
from einops import rearrange

from .base_pipeline import BasePipeline


def get_i2v_mask(lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
    if mask_pixel_values is None:
        msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
    else:
        msk = mask_pixel_values.clone()
    msk[:, :mask_len] = 1
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0] 
    return msk


def _compute_inpaint_weighted_loss(noise_pred, training_target, video_mask, weights, log=False):
    diff = (noise_pred.float() - training_target.float()) ** 2
    inpaint_mask = 1.0 - video_mask
    background_mask = video_mask

    w_inpaint, w_background, w_global = weights
    loss_inpaint_part = diff * inpaint_mask * w_inpaint
    loss_background_part = diff * background_mask * w_background
    loss_global_part = diff * w_global

    total_loss_weight = inpaint_mask * w_inpaint + background_mask * w_background + w_global
    denom = total_loss_weight.sum() + 1e-6
    loss = (loss_inpaint_part + loss_background_part + loss_global_part).sum() / denom

    if log:
        with torch.no_grad():
            print_inpaint = loss_inpaint_part.sum() / denom
            print_background = loss_background_part.sum() / denom
            print_global = loss_global_part.sum() / denom
            raw_mse_inpaint = (diff * inpaint_mask).sum() / (inpaint_mask.sum() + 1e-6)
            raw_mse_background = (diff * background_mask).sum() / (background_mask.sum() + 1e-6)
            print(f"[Loss Info] Step Loss: {loss.item():.6f} | "
                  f"Inpaint Contrib: {print_inpaint.item():.6f} | "
                  f"BG Contrib: {print_background.item():.6f} | "
                  f"Global Contrib: {print_global.item():.6f}")
            print(f"[Raw MSE] Inpaint MSE: {raw_mse_inpaint.item():.6f} | BG MSE: {raw_mse_background.item():.6f}")

    return loss


def apply_face_augmentations(face_tensor, p_aug=0.5):
    """
    对 Face Video 进行数据增强
    Args:
        face_tensor: [B, C, T, H, W], range [-1, 1]
    # 参考 image restoration -> base 3个 distoration -> blur
    """
    # 概率检查
    if torch.rand(1, device=face_tensor.device).item() > p_aug:
        return face_tensor
    
    face_aug = (face_tensor + 1.0) * 0.5
    B, C, T, H, W = face_aug.shape
    device = face_aug.device

    # # 1. 亮度 (Brightness)
    # if torch.rand(1) < 0.5:
    #     brightness_factor = 1.0 + torch.empty(B, 1, 1, 1, 1, device=device).uniform_(-0.2, 0.2)
    #     face_aug = face_aug * brightness_factor

    # # 2. 对比度 (Contrast)
    # if torch.rand(1) < 0.5:
    #     contrast_factor = 1.0 + torch.empty(B, 1, 1, 1, 1, device=device).uniform_(-0.2, 0.2)
    #     mean = face_aug.mean(dim=(2, 3, 4), keepdim=True)
    #     face_aug = (face_aug - mean) * contrast_factor + mean

    # # 1. 亮度 (Brightness)
    # if torch.rand(1) < 0.5:
    #     bright_factor = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.95, 1.05)
    #     face_aug = face_aug * bright_factor

    # # 2. 对比度 (Contrast) - 微调
    # if torch.rand(1) < 0.5:
    #     contrast_factor = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.95, 1.05)
    #     mean = face_aug.mean(dim=(2, 3, 4), keepdim=True)
    #     face_aug = (face_aug - mean) * contrast_factor + mean

    # 3. 饱和度 (Saturation)
    if C == 3 and torch.rand(1) < 0.5:
        sat_factor = 1.0 + torch.empty(B, 1, 1, 1, 1, device=device).uniform_(-0.2, 0.2)
        grayscale = (face_aug[:, 0] * 0.299 + face_aug[:, 1] * 0.587 + face_aug[:, 2] * 0.114)
        grayscale = grayscale.unsqueeze(1).expand_as(face_aug)
        face_aug = grayscale + (face_aug - grayscale) * sat_factor
    
    face_aug = torch.clamp(face_aug, 0.0, 1.0)
    face_aug = face_aug * 2.0 - 1.0

    # 4. 噪声 (Noise)
    if torch.rand(1) < 0.5:
        sigma = torch.empty(1, device=device).uniform_(0.0, 0.05).item()
        noise = torch.randn_like(face_aug) * sigma
        face_aug = face_aug + noise
        face_aug = torch.clamp(face_aug, -1.0, 1.0)

    # 5. 缩放 (Scaling)
    if torch.rand(1) < 0.5:
        scale = torch.empty(1).uniform_(0.8, 1.2).item()
        if abs(scale - 1.0) > 0.01:
            flat_input = rearrange(face_aug, "b c t h w -> (b t) c h w")
            if scale > 1.0: # Zoom In
                crop_h, crop_w = int(H / scale), int(W / scale)
                h_start, w_start = (H - crop_h) // 2, (W - crop_w) // 2
                cropped = flat_input[:, :, h_start:h_start+crop_h, w_start:w_start+crop_w]
                scaled = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
            else: # Zoom Out
                new_h, new_w = int(H * scale), int(W * scale)
                downscaled = F.interpolate(flat_input, size=(new_h, new_w), mode='bilinear', align_corners=False)
                pad_h, pad_w = (H - new_h) // 2, (W - new_w) // 2
                pad_dims = (pad_w, W - new_w - pad_w, pad_h, H - new_h - pad_h)
                scaled = F.pad(downscaled, pad_dims, value=-1.0)
            face_aug = rearrange(scaled, "(b t) c h w -> b c t h w", b=B)

    return face_aug


def FlowMatchSFTLoss_WanAnimate(pipe, **inputs):
    # dropout + augment driving face video 
    inputs = inputs.copy()

    # v1 -> 20260117_110516-DiT_LoRA128-Data_20260115-Ref_Key_Drop -> without dropout
    # p_drop_face = inputs.get("p_drop_face", 0.3)
    # p_drop_bg   = inputs.get("p_drop_bg", 0.1) 
    # p_drop_clip = inputs.get("p_drop_clip", 0.1)

    # v1 -> 20260118_032148-DiT_LoRA128-Data_20260115-Ref_Key_Drop_Aug
    # p_drop_face = inputs.get("p_drop_face", 0.25)
    # p_drop_bg   = inputs.get("p_drop_bg", 0.1) 
    # p_drop_clip = inputs.get("p_drop_clip", 0.0)
    # p_aug = 0.25

    # v2 -> 20260118_112403-DiT_LoRA128-Data_20260115-Ref_Key_Drop_Aug -> 20260121_065044-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug
    # v2 -> without light -> 20260123_124218-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug
    # p_drop_face = inputs.get("p_drop_face", 0.2)
    # p_drop_bg   = inputs.get("p_drop_bg", 0.1) 
    # p_drop_clip = inputs.get("p_drop_clip", 0.0)
    # p_aug = 0.5

    # v3 -> 20260119_210309-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug -> 20260121_064638-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug
    # v3 -> without light -> 20260123_124723-DiT_LoRA128-Data_20260103_19-Ref_Key_Drop_Aug
    p_drop_face = inputs.get("p_drop_face", 0.15)
    p_drop_bg   = inputs.get("p_drop_bg", 0.1) 
    p_drop_clip = inputs.get("p_drop_clip", 0.0)
    p_aug = 0.6

    drop_face = torch.rand(1, device=pipe.device).item() < p_drop_face
    drop_bg   = torch.rand(1, device=pipe.device).item() < p_drop_bg
    drop_clip = torch.rand(1, device=pipe.device).item() < p_drop_clip

    if drop_face and drop_clip and drop_bg:
        resurrect = torch.randint(0, 3, (1,)).item()
        if resurrect == 0: drop_face = False
        elif resurrect == 1: drop_clip = False
        else: drop_bg = False

    # ================= [Part 2: Face Augmentation] =================
    if not drop_face and "face_pixel_values" in inputs:
        inputs["face_pixel_values"] = apply_face_augmentations(
            inputs["face_pixel_values"], 
            p_aug=p_aug
        )
        inputs["face_pixel_values"] = inputs["face_pixel_values"].to(dtype=pipe.torch_dtype)

    # ================= [Part 3: Apply Dropout] =================
    # 1. Face Dropout
    if drop_face and "face_pixel_values" in inputs:
        inputs["face_pixel_values"] = torch.zeros_like(
            inputs["face_pixel_values"], 
            dtype=pipe.torch_dtype
        )
    
    # 2. CLIP Dropout
    if drop_clip and "clip_feature" in inputs:
        inputs["clip_feature"] = torch.zeros_like(inputs["clip_feature"], dtype=pipe.torch_dtype)

    # 3. Background Dropout (Dynamic split of 'y')
    if drop_bg and "y" in inputs:
        y_tensor = inputs["y"]
        C = y_tensor.shape[1]
        if C > 0 and C % 2 == 0:
            split_idx = C // 2
            y_new = y_tensor.clone()
            y_new[:, split_idx:] = 0 # 置零 Background Latents
            inputs["y"] = y_new

    if "face_pixel_values" in inputs and inputs["face_pixel_values"].dtype != pipe.torch_dtype:
        inputs["face_pixel_values"] = inputs["face_pixel_values"].to(dtype=pipe.torch_dtype)

    # ================= Loss Forward =================
    max_t_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_t_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_t_boundary, max_t_boundary, (1,)) 
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep) 

    # Mask
    mask_pixels = 1 - pipe.preprocess_video(inputs["animate_mask_video"], max_value=1, min_value=0)
    lat_t = noise_pred.shape[2] 
    target_raw_frames = (lat_t - 1) * 4 + 1

    if mask_pixels.shape[2] < target_raw_frames:
        num_pad = target_raw_frames - mask_pixels.shape[2]
        padding = mask_pixels[:, :, -1:].repeat(1, 1, num_pad, 1, 1)
        mask_pixels = torch.cat([mask_pixels, padding], dim=2)
    
    _, _, _, lat_h, lat_w = noise_pred.shape
    mask_pixels_tmp = rearrange(mask_pixels, "b c t h w -> (b t) c h w")
    mask_pixels_tmp = F.interpolate(mask_pixels_tmp, size=(lat_h, lat_w), mode='nearest')
    mask_pixels_lat = rearrange(mask_pixels_tmp, "(b t) c h w -> b t c h w", b=1)[:, :, 0]

    full_mask = get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixels_lat, pipe.device).to(dtype=noise_pred.dtype) 
    
    video_mask = full_mask[0] if full_mask.shape[0] == 4 else full_mask[:, 0]
    if video_mask.shape[1] != noise_pred.shape[3]:
        video_mask = video_mask.transpose(1, 2)
    video_mask = video_mask.unsqueeze(0).unsqueeze(0)

    # Loss
    loss_weights = (1.0, 0.2, 0.2)
    loss = _compute_inpaint_weighted_loss(
        noise_pred, training_target, video_mask, loss_weights, log=False
    )
    
    # Conditional Loss Scaling
    cond_scale = 1.0
    if drop_bg:
        cond_scale *= 2.0 
    if drop_face:
        cond_scale *= 1.1

    loss = loss * cond_scale
    loss = loss * pipe.scheduler.training_weight(timestep)
    
    return loss


def FlowMatchSFTLoss_WanAnimate2(pipe, **inputs):
    # dropout + augment driving face video 
    inputs = inputs.copy()

    p_drop_face = inputs.get("p_drop_face", 0.15)
    p_drop_bg   = inputs.get("p_drop_bg", 0.1) 
    p_drop_clip = inputs.get("p_drop_clip", 0.0)
    p_aug = 0.6

    drop_face = torch.rand(1, device=pipe.device).item() < p_drop_face
    drop_bg   = torch.rand(1, device=pipe.device).item() < p_drop_bg
    drop_clip = torch.rand(1, device=pipe.device).item() < p_drop_clip

    if drop_face and drop_clip and drop_bg:
        resurrect = torch.randint(0, 3, (1,)).item()
        if resurrect == 0: drop_face = False
        elif resurrect == 1: drop_clip = False
        else: drop_bg = False

    # ================= [Part 2: Face Augmentation] =================
    if not drop_face and "face_pixel_values" in inputs:
        inputs["face_pixel_values"] = apply_face_augmentations(
            inputs["face_pixel_values"], 
            p_aug=p_aug
        )
        inputs["face_pixel_values"] = inputs["face_pixel_values"].to(dtype=pipe.torch_dtype)

    # ================= [Part 3: Apply Dropout] =================
    # 1. Face Dropout
    if drop_face and "face_pixel_values" in inputs:
        inputs["face_pixel_values"] = torch.zeros_like(
            inputs["face_pixel_values"], 
            dtype=pipe.torch_dtype
        )
    
    # 2. CLIP Dropout
    if drop_clip and "clip_feature" in inputs:
        inputs["clip_feature"] = torch.zeros_like(inputs["clip_feature"], dtype=pipe.torch_dtype)

    # 3. Background Dropout (Dynamic split of 'y')
    if drop_bg and "y" in inputs:
        y_tensor = inputs["y"]
        C = y_tensor.shape[1]
        if C > 0 and C % 2 == 0:
            split_idx = C // 2
            y_new = y_tensor.clone()
            y_new[:, split_idx:] = 0 # 置零 Background Latents
            inputs["y"] = y_new

    if "face_pixel_values" in inputs and inputs["face_pixel_values"].dtype != pipe.torch_dtype:
        inputs["face_pixel_values"] = inputs["face_pixel_values"].to(dtype=pipe.torch_dtype)

    # ================= Loss Forward =================
    max_t_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_t_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_t_boundary, max_t_boundary, (1,)) 
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep) 

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    
    return loss