import torch, os, argparse, accelerate, warnings, json
from PIL import Image
from wan_parser import get_wan_base_parser
from diffsynth.core import UnifiedDataset2
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.diffusion import *
from diffsynth.diffusion.runner_fsdp import get_fsdp_plugin  # explicitly import for type checkers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        use_ref_as_input=False,
        ref_image_key="input_image",
        face_model_path=None,
        use_usp=False
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Training Pipeline
        if args.pipeline_processor in ['wan_video']:
            print("Loading Default Wan Pipeline")
            from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
        elif args.pipeline_processor in ['wan_video_face']:
            print("Loading Wan-Animate Arcface Pipeline")
            from diffsynth.pipelines.wan_video_face import WanVideoPipeline, ModelConfig
        else:
            raise ValueError(f"Unsupported pipeline processor: {args.pipeline_processor}")
        
        # Load models
        model_configs = self.parse_model_configs(
            model_paths, 
            model_id_with_origin_paths, 
            fp8_models=fp8_models, 
            offload_models=offload_models, 
            device=device
        )
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        if audio_processor_path is None:
            audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") 
        else:
            ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device=device, 
            model_configs=model_configs, 
            tokenizer_config=tokenizer_config, 
            audio_processor_config=None,
            use_usp=use_usp
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            # default training strategy
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            # Wan Animate
            "sft:train2": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train3": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:wan_animate": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss_WanAnimate(pipe, **inputs_shared, **inputs_posi),
            # "sft:train_wan_animate": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss_WanAnimate(pipe, **inputs_shared, **inputs_posi),
            "sft:train_wan_animate": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss_WanAnimate2(pipe, **inputs_shared, **inputs_posi),
            "sft:train3_wan_animate": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss_WanAnimate(pipe, **inputs_shared, **inputs_posi),
            "sft:fsdp_train_wan_animate": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss_WanAnimate(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.use_ref_as_input = use_ref_as_input
        self.ref_image_key = ref_image_key

        self._print_model_param_counts()

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                first_frame = data["video"][0]
                if self.use_ref_as_input:
                    ref_img = data[self.ref_image_key][0]
                    if isinstance(ref_img, str): ref_img = Image.open(ref_img).convert("RGB")
                    inputs_shared["input_image"] = ref_img.resize(first_frame.size, Image.BICUBIC)
                else:
                    inputs_shared["input_image"] = first_frame
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None, load_metadata_and_cache=False):
        if load_metadata_and_cache and inputs is not None:
            if isinstance(inputs[0], dict): 
                inputs[0].update({k: v for k, v in data.items() if k not in inputs[0]})
            units_to_run = self.pipe.units + self.pipe.face_units   # need to change
        else:
            if inputs is None: 
                inputs = self.get_pipeline_inputs(data)
            units_to_run = self.pipe.units

        # Ensure rand_device is set correctly without assuming latents already exist.
        # If latents are already prepared (e.g. from cached samples), align rand_device
        # with their device; otherwise, fall back to the pipeline device.
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0 and isinstance(inputs[0], dict):
            if 'latents' in inputs[0]:
                inputs[0]['rand_device'] = inputs[0]['latents'].device
            else:
                inputs[0]['rand_device'] = self.pipe.device

        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in units_to_run:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss
    
    def _format_param_count(self, param_count: int) -> str:
        if param_count >= 1_000_000_000:
            return f"{param_count / 1_000_000_000:.3f}B"
        return f"{param_count / 1_000_000:.3f}M"

    def _count_module_params(self, module, trainable_only: bool = False) -> int:
        params = module.parameters()
        if trainable_only:
            params = filter(lambda p: p.requires_grad, params)
        return sum(p.numel() for p in params)

    def _print_model_param_counts(self):
        total_params = self._count_module_params(self.pipe)
        trainable_params = self._count_module_params(self.pipe, trainable_only=True)
        print(f"Model total parameters: {self._format_param_count(total_params)}")
        print(f"Model trainable parameters: {self._format_param_count(trainable_params)}")
        if self.pipe.dit is not None:
            dit_params = self._count_module_params(self.pipe.dit)
            print(f"self.pipe.dit total parameters: {self._format_param_count(dit_params)}")


def wan_parser():
    base_parser = get_wan_base_parser()
    parser = argparse.ArgumentParser(description="My Custom Training Script extending Wan", parents=[base_parser],formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--pipeline_processor", type=str, default="wan_video", choices=["wan_video", "wan_video_face"], help="Pipeline processor to use. wan_video: Wan2.2-Animate-14B, wan_video_face: Wan2.2-Animate-Arcface-14B.")

    parser.add_argument("--use_ref_as_input", action="store_true", help="Use a specific reference image as the model's first-frame input.")
    parser.add_argument("--ref_image_key", type=str, default="animate_ref_identity_image", help="The key/column name in the dataset containing the reference image. e.g. animate_ref_identity_standing")

    parser.add_argument("--use_unified_sequence_parallel", action="store_true",)
    parser.add_argument("--advanced_parallel_strategy", type=str, default="ddp", choices=["fsdp", "fsdp_v2", "fsdp_usp", "ddp"],)
    parser.add_argument("--fsdp_target_models", type=str, default="dit",)

    parser.add_argument("--load_metadata_and_cache", action="store_true")
    parser.add_argument("--load_base_and_added_cache", action="store_true")
    parser.add_argument("--dataset_added_cache_path", type=str, default=None)

    parser.add_argument("--face_model_path", type=str, default=None, help="path refer to : /mnt/beegfs/mingyang/Human-Replacement")

    return parser.parse_args()



if __name__ == "__main__":
    args = wan_parser()
    fsdp_plugin, kwargs_handlers= None, []
    if args.advanced_parallel_strategy in ["fsdp", "fsdp_usp"]:
        fsdp_plugin = get_fsdp_plugin(args.fsdp_target_models)
    elif args.advanced_parallel_strategy in ["ddp"]:
        kwargs_handlers.append(accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters))
    accelerator = accelerate.Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin = fsdp_plugin, kwargs_handlers=kwargs_handlers,
    )
    dataset = UnifiedDataset2(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset2.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        },
        load_metadata_and_cache=args.load_metadata_and_cache,
        load_base_and_added_cache=args.load_base_and_added_cache,
        added_cache_path=args.dataset_added_cache_path,
    )
    # Validation subset by stable id (clip_id / path), so same samples every step regardless of DataLoader shuffle.
    val_dataset = None
    if getattr(args, "val_num_videos", 0) and args.val_num_videos > 0:
        from torch.utils.data import Subset
        num_val = min(args.val_num_videos, len(dataset))
        val_ids_path = os.path.join(args.output_path, "val", "val_ids.json") if getattr(args, "output_path", None) else None
        val_ids = None
        if val_ids_path and os.path.isfile(val_ids_path):
            try:
                with open(val_ids_path, "r") as f:
                    saved = json.load(f)
                raw = saved.get("val_ids", [])
                val_ids = [tuple(x) if isinstance(x, list) else x for x in raw]
                if accelerator.is_main_process and val_ids:
                    accelerator.print(f"[VAL] Reusing validation ids from {val_ids_path} (count={len(val_ids)})")
            except Exception:
                pass
        if not val_ids:
            seen = set()
            val_ids = []
            for i in range(len(dataset)):
                sid = dataset.get_stable_id(i)
                if sid not in seen:
                    seen.add(sid)
                    val_ids.append(sid)
                if len(val_ids) >= num_val:
                    break
        val_indices = dataset.get_indices_for_ids(val_ids)[:num_val]
        if val_ids_path and accelerator.is_main_process:
            os.makedirs(os.path.dirname(val_ids_path), exist_ok=True)
            with open(val_ids_path, "w") as f:
                json.dump({"val_ids": [list(s) if isinstance(s, tuple) else s for s in val_ids]}, f, indent=2)
        val_dataset = Subset(dataset, val_indices)

    # Multi-GPU: only main process downloads first to avoid concurrent ModelScope requests (ChunkedEncodingError).
    # Other ranks wait, then create model from local cache. No change to core/loader.
    _model_kw = dict(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        use_ref_as_input=args.use_ref_as_input,
        ref_image_key=args.ref_image_key,
        face_model_path=args.face_model_path,
        use_usp=args.use_unified_sequence_parallel,
    )
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            model = WanTrainingModule(**_model_kw)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            model = WanTrainingModule(**_model_kw)
    else:
        model = WanTrainingModule(**_model_kw)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        # default mapping
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        # # Wan Animate
        # "sft:train2": launch_training_task2,            # add load_metadata_and_cache: load cached data and metadata rows; 训练过程中，先load一些数据，再根据metadata更新新的inputs
        # "sft:train3": launch_training_task3,            # load base cached_data and new cached_data
        # "sft:wan_animate": launch_training_task,
        # "sft:train_wan_animate": launch_training_task,
        # "sft:train3_wan_animate": launch_training_task3,
        # # in developing
        # "sft:fsdp_train_wan_animate": launch_fsdp_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args, val_dataset=val_dataset)
