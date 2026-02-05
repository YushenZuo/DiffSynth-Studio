import os, torch, gc
from accelerate import Accelerator


# class ModelLogger:
#     def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
#         self.output_path = output_path
#         self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
#         self.state_dict_converter = state_dict_converter
#         self.num_steps = 0


#     def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
#         self.num_steps += 1
#         if save_steps is not None and self.num_steps % save_steps == 0:
#             self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


#     def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
#         accelerator.wait_for_everyone()
#         if accelerator.is_main_process:
#             state_dict = accelerator.get_state_dict(model)
#             state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
#             state_dict = self.state_dict_converter(state_dict)
#             os.makedirs(self.output_path, exist_ok=True)
#             path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
#             accelerator.save(state_dict, path, safe_serialization=True)


#     def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
#         if save_steps is not None and self.num_steps % save_steps != 0:
#             self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


#     def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
#         accelerator.wait_for_everyone()
#         if accelerator.is_main_process:
#             state_dict = accelerator.get_state_dict(model)
#             state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
#             state_dict = self.state_dict_converter(state_dict)
#             os.makedirs(self.output_path, exist_ok=True)
#             path = os.path.join(self.output_path, file_name)
#             accelerator.save(state_dict, path, safe_serialization=True)


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        checkpoint_extension=".safetensors",
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.current_epoch = None
        self.checkpoint_extension = checkpoint_extension
        self._safe_serialization = checkpoint_extension.endswith(".safetensors")
        self.optimizer = None
        self.scheduler = None

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, use_fsdp_v1=False):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            file_name = f"step-{self.num_steps}" if use_fsdp_v1 else f"step-{self.num_steps}{self.checkpoint_extension}"
            self.save_model(accelerator, model, file_name, use_fsdp_v1=use_fsdp_v1)

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id, use_fsdp_v1=False):
        self.current_epoch = epoch_id
        file_name = f"epoch-{epoch_id}" if use_fsdp_v1 else f"epoch-{epoch_id}{self.checkpoint_extension}"
        self.save_model(accelerator, model, file_name, use_fsdp_v1=use_fsdp_v1)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, use_fsdp_v1=False):
        if save_steps is not None and self.num_steps % save_steps != 0:
            file_name = f"step-{self.num_steps}" if use_fsdp_v1 else f"step-{self.num_steps}{self.checkpoint_extension}"
            self.save_model(accelerator, model, file_name, use_fsdp_v1=use_fsdp_v1)

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name, use_fsdp_v1=False):
        accelerator.wait_for_everyone()
        if use_fsdp_v1:
            path = os.path.join(self.output_path, file_name)
            if accelerator.is_main_process: os.makedirs(path, exist_ok=True)
            accelerator.wait_for_everyone()
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            accelerator.print(f"[INFO] Start saving FSDP sharded state to {path}.")
            # accelerator.save_state(path)
            accelerator.save_model(model, path, safe_serialization=self._safe_serialization)    # only save checkpoints
            accelerator.print(f"[INFO] FSDP Sharded checkpoint saved successfully.")
            return
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            self._save_checkpoint(accelerator, path, state_dict)

    def attach_training_components(self, optimizer=None, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _build_checkpoint(self, model_state_dict):
        if self._safe_serialization:
            checkpoint = dict(model_state_dict)
            checkpoint["_meta_step"] = torch.tensor(self.num_steps, dtype=torch.int64)
            if self.current_epoch is not None:
                checkpoint["_meta_epoch"] = torch.tensor(self.current_epoch, dtype=torch.int64)
            return checkpoint
        checkpoint = {"model": model_state_dict, "step": self.num_steps}
        if self.current_epoch is not None:
            checkpoint["epoch"] = self.current_epoch
        if self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        return checkpoint

    def _save_checkpoint(self, accelerator, path, model_state):
        checkpoint = self._build_checkpoint(model_state)
        accelerator.save(checkpoint, path, safe_serialization=self._safe_serialization)