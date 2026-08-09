import json
import os
from collections.abc import Mapping

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor
from trl import get_kbit_device_map, get_quantization_config

from latsearch.reward.config import (
    ModelConfig,
    PEFTLoraConfig,
    TrainingConfig,
    load_model_from_checkpoint,
)
from latsearch.reward.data import DataConfig
from latsearch.reward.head import Qwen2VLRewardModelBT
from latsearch.reward.prompts import build_prompt
from latsearch.reward.vision import process_vision_info


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):
    """
    Find the target linear modules for LoRA.
    """
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue

        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def create_model_and_processor(model_config, peft_lora_config, training_args, cache_dir=None):
    # create model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=True if training_args.gradient_checkpointing else False,
    )

    # create processor and set padding
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right", cache_dir=cache_dir
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2"
        if not training_args.disable_flash_attn2
        else "sdpa",
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,
        **model_kwargs,
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    # create lora and peft model
    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude,
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config


def load_configs_from_json(config_path):
    with open(config_path) as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return (
        config_dict["data_config"],
        None,
        config_dict["model_config"],
        config_dict["peft_lora_config"],
        config_dict["inference_config"] if "inference_config" in config_dict else None,
    )


class LatentReward(nn.Module):
    def __init__(
        self,
        load_from_pretrained,
        load_from_pretrained_step=-1,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(
            config_path
        )
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)
        self.device = device

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )

        model, processor, _ = create_model_and_processor(
            model_config, peft_lora_config, training_args
        )
        model, _ = load_model_from_checkpoint(
            model, load_from_pretrained, load_from_pretrained_step
        )

        # Manually replace patch_embed.proj with new Conv3d layer and reinitialize it
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        visual_module = getattr(base_model, "visual", None)
        if visual_module is None:
            visual_module = base_model.model.visual
        patch_embed = visual_module.patch_embed
        old_proj = patch_embed.proj
        patch_embed.proj = nn.Conv3d(
            in_channels=16,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=False,
        ).to(dtype=dtype, device=device)
        patch_embed.in_channels = 16
        torch.nn.init.xavier_uniform_(patch_embed.proj.weight)

        # Latents are already in the model's feature space, so RGB image
        # preprocessing must remain disabled.
        video_proc = getattr(processor, "video_processor", processor.image_processor)
        video_proc.do_convert_rgb = False
        video_proc.do_rescale = False
        video_proc.do_normalize = False
        video_proc.image_mean = [0.0] * 16
        video_proc.image_std = [1.0] * 16

        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = True

        self.model = model
        self.processor = processor
        self.video_processor = video_proc
        self.data_config = data_config
        self.inference_config = inference_config
        # The paper conditions every latent on a learned denoising-step
        # embedding. Keep it on the module so it is optimized and checkpointed.
        self.step_embedding = nn.Embedding(1000, 16, device=device, dtype=dtype)
        self.to(self.device)

    def forward(self, batch):
        return self.model(return_dict=True, **batch)["logits"]

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch(
        self, videos, prompts, denoising_steps, fps=None, num_frames=None, max_pixels=None
    ):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video,
                                "max_pixels": max_pixels,
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompt,
                                    self.data_config.eval_dim,
                                    self.data_config.prompt_template_type,
                                ),
                            },
                        ],
                    },
                ]
                for video, prompt in zip(videos, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video,
                                "max_pixels": max_pixels,
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompt,
                                    self.data_config.eval_dim,
                                    self.data_config.prompt_template_type,
                                ),
                            },
                        ],
                    },
                ]
                for video, prompt in zip(videos, prompts)
            ]

        image_inputs, video_inputs = process_vision_info(chat_data)
        video_inputs = torch.stack(video_inputs, dim=0)  # [B, T, C, H, W]

        B, T, C, H, W = video_inputs.shape
        if C != self.step_embedding.embedding_dim:
            raise ValueError(
                f"Expected {self.step_embedding.embedding_dim} latent channels, got {C}."
            )
        # Hugging Face image processors convert tensors to NumPy internally,
        # which requires CPU float32 rather than CUDA bfloat16.
        video_inputs = [video_inputs[i].float().cpu() for i in range(B)]

        batch = self.processor(
            text=self.processor.apply_chat_template(
                chat_data, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={
                "do_resize": False,
                "do_convert_rgb": False,
                "do_rescale": False,
                "do_normalize": False,
                "input_data_format": "channels_first",
            },
        )
        batch = self._prepare_inputs(batch)

        # Add the channel-wise timestep embedding after preprocessing. Adding
        # it earlier would lose its gradient when the processor converts video
        # tensors to NumPy arrays.
        patch_volume = self.video_processor.temporal_patch_size * self.video_processor.patch_size**2
        step_patches = self.step_embedding(denoising_steps.long())
        step_patches = step_patches.to(batch["pixel_values_videos"].dtype)
        step_patches = step_patches.repeat_interleave(patch_volume, dim=1)
        patch_counts = batch["video_grid_thw"].prod(dim=1)
        step_patches = torch.repeat_interleave(step_patches, patch_counts, dim=0)
        if step_patches.shape != batch["pixel_values_videos"].shape:
            raise ValueError(
                "Timestep embedding shape does not match the processed latent patches: "
                f"{step_patches.shape} != {batch['pixel_values_videos'].shape}."
            )
        batch["pixel_values_videos"] = batch["pixel_values_videos"] + step_patches
        return batch
