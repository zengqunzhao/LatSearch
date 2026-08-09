import gc
import logging
import math
import os
import types
from contextlib import contextmanager
from functools import partial
import torch
from torch import amp
import torch.distributed as dist
from tqdm import tqdm
import time
import random
from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torch.fft as fft
import torch.nn.functional as FF
import copy
import numpy as np
from .utils.freeinit_utils import (
    get_freq_filter,
    freq_mix_3d,
)
from .utils.freqprior_utils import (
    get_freq_filter,
    freq_mix_noise_3d,
)


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt


    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 time_dic=None
        ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        if time_dic is None:
            time_dic = {"DiT_Time": [], "Decoder_Time": []}
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    algorithm_type='sde-dpmsolver++',
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            start_time = time.time()
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            print(f"DiT Time: {(time.time() - start_time):.2f} sec")
            time_dic["DiT_Time"].append(time.time()-start_time)

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            start_time = time.time()
            if self.rank == 0:
                videos = self.vae.decode(x0)
            print(f"Decoder Time: {(time.time() - start_time):.2f} sec")
            time_dic["Decoder_Time"].append(time.time()-start_time)

            # if self.rank == 0:
            #     videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


    def generate_with_latent(self,
                    input_prompt,
                    size=(1280, 720),
                    frame_num=81,
                    shift=5.0,
                    sample_solver='unipc',
                    sampling_steps=50,
                    guide_scale=5.0,
                    n_prompt="",
                    seed=-1,
                    offload_model=True,
                    selected_timestep_keys=[],
                    ):

        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    algorithm_type='sde-dpmsolver++',
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            stored_latents = {}

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                # Store latents at selected steps
                if i in selected_timestep_keys:  # when use timesteps, absolute t is needed
                    stored_latents[i] = latents[0].clone().detach()

            x0 = latents

            similarities = {
                t: torch.nn.functional.cosine_similarity(x0[0].flatten(), stored_latents[t].flatten(), dim=0).item()
                for t in selected_timestep_keys
                }


            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None, stored_latents, similarities


    def generate_N_videos(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='dpm++',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 number_of_N=6,
                 time_dic=None
        ):
        """
        Generates video frames from text prompt using diffusion process.
        """
        if time_dic is None:
            time_dic = {"DiT_Time": [], "Decoder_Time": []}

        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        num_variants = number_of_N - 1
        noise_parent = torch.randn(
            1,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g
        )
        epsilons = torch.randn(
            num_variants,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g
        )
        noise_candidates = torch.cat([noise_parent, epsilons], dim=0)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        start_time = time.time()
        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(number_of_N):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(number_of_N):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            latents_total = noise_candidates

            for j, t in enumerate(tqdm(timesteps)):

                latents_list = []

                for k in range(number_of_N):

                    latents = latents_total[k]
                    latent_model_input = latents[None]

                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    latents_, _ = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)
                    latents_list.append(latents_)
                latents_total=torch.cat(latents_list)
            x0 = latents_total

            print(f"DiT Time: {(time.time() - start_time):.2f} sec")
            time_dic["DiT_Time"].append(time.time()-start_time)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            start_time = time.time()
            if self.rank == 0:
                videos = self.vae.decode(x0)
            print(f"Decoder Time: {(time.time() - start_time):.2f} sec")
            time_dic["Decoder_Time"].append(time.time()-start_time)

        del noise_candidates, latents_total
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos if self.rank == 0 else None

    def generate_get_reward(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 verifier_video=None,
                 verifier_latent=None
        ):

        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        STEP_TO_TIMESTEP =  {
                0 : 0,
                5 : 100,
                10: 200,
                15: 300,
                20: 400,
                25: 500,
                30: 600,
                35: 700,
                40: 800
            }

        reward_scheduler = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        latent_reward_dict = {}
        video_reward_dict = {}

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    algorithm_type='sde-dpmsolver++',
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]
                timestep = torch.stack(timestep)
                self.model.to(self.device)
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                if idx in reward_scheduler:

                    # ---------- Latent Reward ----------
                    if verifier_latent is not None:
                        latent = latents[0]  # (C, T, H, W)
                        latent_video = latent.permute(1, 0, 2, 3).contiguous()  # (T, C, H, W)
                        n_frames = latent_video.shape[0]
                        verifier_tensors = [latent_video]
                        verifier_prompt = [input_prompt]
                        verifier_steps = torch.full(
                            (1,), STEP_TO_TIMESTEP[idx],
                            device=self.device,
                            dtype=torch.long
                        )

                        verifier_inputs = verifier_latent.prepare_batch(
                            videos=verifier_tensors,
                            prompts=verifier_prompt,
                            denoising_steps=verifier_steps,
                            num_frames=n_frames,
                        )
                        latent_rewards = verifier_latent(verifier_inputs).sum(dim=1)  # (1,)
                        latent_reward_dict[idx] = latent_rewards.detach().cpu().item()

                    # ---------- Video Reward ----------
                    if verifier_video is not None:
                        decoded_list = self.vae.decode(latents)    # list of (T, C, H, W)
                        # The decoder normally returns a list; this call has one candidate.
                        if isinstance(decoded_list, (list, tuple)):
                            video_tensor = decoded_list[0]         # (T, C, H, W)
                        else:
                            video_tensor = decoded_list            # Handle a direct (T, C, H, W) tensor.

                        video_batch = video_tensor.unsqueeze(0)    # (1, T, C, H, W)
                        rewards = verifier_video.reward(
                            video_batch.permute(0, 2, 1, 3, 4),    # (1, C, T, H, W)
                            [input_prompt],
                            use_norm=False
                        )
                        overall_scores = torch.tensor(
                            [r['Overall'] for r in rewards],
                            device=self.device
                        )
                        video_reward_dict[idx] = overall_scores.detach().cpu().item()

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)  # list of (T, C, H, W)
                if isinstance(videos, (list, tuple)):
                    video_out = videos[0]
                else:
                    video_out = videos
            else:
                video_out = None

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return video_out, latent_reward_dict, video_reward_dict


class WanT2VWithEvoSearch:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        reward_model=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.reward_model= reward_model
        self.model.eval().requires_grad_(False)
        # self.reward_model.eval().requires_grad_(False)
        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def evosearch(self,
                 input_prompt,
                 ## EvoSearch args ##
                 elite_size: int = 3,
                 generation_steps: int = 0,
                 mutation_rate: float = 0.2,
                 evolution_schedule=None,
                 population_size_schedule=None,
                 verifier=None,
                 ###
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 number_of_N=1,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 noise=None,
                 generator=None,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        if noise is None:
            target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        else:
            target_shape = noise[0].shape
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt


        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]


        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        # with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():
            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(number_of_N):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(number_of_N):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            ## EvoSearch
            latent_to_decode_fn=lambda x: latent_to_decode(vae=self.vae, latent=x)
            generation_steps_id = generation_steps
            std_list=[-1 for _ in evolution_schedule]
            current_step = evolution_schedule[generation_steps]
            if noise is None:
                noise = [
                    torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    generator=generator).to(self.device)
                        ]
            latents_total=noise

            # Track the three runtime components within EvoSearch.
            denoise_time_evo = 0.0      # DiT denoising time.
            decoder_time_evo = 0.0      # Population decoding time.
            reward_time_evo  = 0.0      # Reward evaluation time.

            for j, t in zip(range(current_step,sampling_steps),timesteps[current_step:]):
                latents_list = []

                step_start = time.time()

                for k in range(number_of_N):
                    latents = latents_total[k]
                    latent_model_input = latents[None]
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    latents_, _, variance, std = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=generator)
                    latents_list.append(latents_)
                    if j in evolution_schedule:
                        self.population_list[generation_steps_id].append(latents_)
                        self.variance_list[generation_steps_id].append(variance)
                latents_total = torch.cat(latents_list)

                denoise_time_evo += time.time() - step_start

                if j in evolution_schedule:
                    std_list[generation_steps_id]=std
                    generation_steps_id += 1

            reward_list=[]
            for k in range(number_of_N):
                # Measure decoder time.
                torch.cuda.synchronize()
                dec_start = time.time()
                population_video = latent_to_decode_fn(latents_total[k])[0]
                torch.cuda.synchronize()
                decoder_time_evo += time.time() - dec_start
                self.video_list.append(population_video)
                # Measure reward evaluation time.
                torch.cuda.synchronize()
                rew_start = time.time()
                reward = verifier.reward([population_video.permute(1,0,2,3)], [input_prompt], use_norm=True)
                torch.cuda.synchronize()
                reward_time_evo += time.time() - rew_start
                reward_list.append(reward[0]['Overall'])
            rewards = torch.tensor(reward_list).to(self.device)
            for id in range(generation_steps,len(self.rewards_list)):
                self.rewards_list[id].append(rewards)
            population = torch.cat(self.population_list[generation_steps])

            mean_std=std_list[generation_steps]
            variance= torch.cat(self.variance_list[generation_steps])
            rewards = torch.cat(self.rewards_list[generation_steps])
            elite_rewards= rewards
            elite_rew, elite_indices = torch.topk(elite_rewards, elite_size)
            if elite_rew[0]>self.best_reward:
                self.best_reward = elite_rew[0]
                ind = elite_indices[0]
                self.best_video = self.video_list[ind]

            elites = population[elite_indices]
            parents = []
            population_size= population_size_schedule[generation_steps+1]
            for _ in range(population_size-elite_size):
                candidates = torch.randperm(population.shape[0])[:int(population.shape[0]*0.9)]
                candidate_rewards = rewards[candidates]
                winner = candidates[torch.argmax(candidate_rewards)]
                parents.append(population[winner])
            if not parents:
                return elites, denoise_time_evo, decoder_time_evo, reward_time_evo

            parents = torch.stack(parents)
            if generation_steps == 0:
                children = parents * math.sqrt(1 - mutation_rate**2) + mutation_rate * torch.randn_like(parents)
            else:
                children = parents  + mean_std * torch.randn_like(parents)
            children = torch.cat([elites,children])
            return children, denoise_time_evo, decoder_time_evo, reward_time_evo

    def generate(self,
                 input_prompt,
                 ## EvoSearch args ##
                 elite_size: int = 3,
                 guidance_reward: str='VideoReward',
                 mutation_rate: float = 0.2,
                 evolution_schedule=None,
                 population_size_schedule=None,
                 verifier=None,
                 ###
                 size=(1280, 720),
                 frame_num=33,
                 shift=5.0,
                 number_of_N=1,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 noise=None,
                 generator=None,
                 offload_model=True,
                 time_dic=None
        ):
        if time_dic is None:
            time_dic = {"DiT_Time": [], "Decoder_Time": [], "Reward_Time": []}
        # preprocess
        F = frame_num
        if noise is None:
            target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        else:
            target_shape = noise[0].shape
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt


        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():
            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(number_of_N):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(number_of_N):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            ## Initialization
            self.best_reward=-100
            self.best_video=None
            self.video_list=[]
            generation_steps = 0

            self.population_list = [[] for _ in evolution_schedule]
            self.rewards_list = [[] for _ in evolution_schedule]
            self.variance_list = [[] for _ in evolution_schedule]
            if noise is None:
                noise = [
                    torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    generator=generator).to(self.device)
                        ]
            latents_total=noise

            # Global timers include both standard sampling and EvoSearch.
            total_denoise_time = 0.0
            total_decoder_time = 0.0
            total_reward_time  = 0.0

            for j, t in enumerate(tqdm(timesteps)):
                latents_list = []
                if j in evolution_schedule:
                    evo_start_time = time.time()

                    latents_total, denoise_evo, decode_evo, reward_evo = self.evosearch(
                        input_prompt=input_prompt,
                        n_prompt=n_prompt,
                        frame_num=frame_num,
                        guide_scale=guide_scale,
                        noise=latents_total,
                        seed=seed,
                        number_of_N=number_of_N,
                        sample_solver=sample_solver,
                        evolution_schedule=evolution_schedule,
                        population_size_schedule=population_size_schedule,
                        elite_size=elite_size,
                        mutation_rate=mutation_rate,
                        verifier=verifier,
                        generator=generator,
                        offload_model=False,
                        )

                    # Add the internal EvoSearch timings to the totals.
                    total_denoise_time += denoise_evo
                    total_decoder_time += decode_evo
                    total_reward_time  += reward_evo

                    generation_steps += 1
                    number_of_N = len(latents_total)
                    sample_schedulers = []
                    for _ in range(number_of_N):
                        sample_scheduler = FlowDPMSolverMultistepScheduler(
                            num_train_timesteps=self.num_train_timesteps,
                            shift=1,
                            algorithm_type='sde-dpmsolver++',
                            use_dynamic_shifting=False)
                        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                        timesteps, _ = retrieve_timesteps(
                            sample_scheduler,
                            device=self.device,
                            sigmas=sampling_sigmas)
                        sample_schedulers.append(sample_scheduler)

                # Measure denoising time for this outer sampling step.
                torch.cuda.synchronize()
                sample_start = time.time()

                for k in range(number_of_N):
                    latents = latents_total[k]
                    latent_model_input = latents[None]
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    latents_, _, _, _ = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=generator)
                    latents_list.append(latents_)
                latents_total=torch.cat(latents_list)

                torch.cuda.synchronize()
                sample_duration = time.time() - sample_start
                total_denoise_time += sample_duration

            x0 = latents_total
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            # Measure VAE decoding time.
            torch.cuda.synchronize()
            decode_start = time.time()
            if self.rank == 0:
                videos =[]
                for item in x0:
                    videos.append(self.vae.decode(item.unsqueeze(0))[0])
                videos=torch.stack(videos)
            torch.cuda.synchronize()
            decode_duration = time.time() - decode_start
            total_decoder_time += decode_duration

        # Measure the final batched reward evaluation.
        torch.cuda.synchronize()
        reward_start = time.time()
        with torch.no_grad():
            rewards = verifier.reward(videos.permute(0,2,1,3,4),
                                      [input_prompt]*videos.shape[0],
                                      use_norm=True)
        torch.cuda.synchronize()
        reward_duration = time.time() - reward_start
        total_reward_time += reward_duration

        rewards = torch.tensor([rewards[i]['Overall'] for i in range(len(rewards))]).to(self.device)
        elite_rew, elite_indices = torch.topk(rewards, 1)
        if elite_rew[0]>self.best_reward:
            self.best_reward = elite_rew[0]
            ind = elite_indices[0]
            self.best_video = videos[ind]
        # Offload all models
        # print('Updated best reward',self.best_reward)
        del noise, latents
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # Store all three pipeline timing components.
        time_dic["DiT_Time"].append(total_denoise_time)
        time_dic["Decoder_Time"].append(total_decoder_time)
        time_dic["Reward_Time"].append(total_reward_time)

        print(f"DiT Time (all): {total_denoise_time:.2f} sec")
        print(f"Decoder Time (all): {total_decoder_time:.2f} sec")
        print(f"Reward Time (all): {total_reward_time:.2f} sec")

        return (self.best_video,)


def latent_to_decode(vae,latent):
    return vae.decode(latent[None])


class WanT2VWithLatSearch:
    """Wan2.1 text-to-video pipeline with LatSearch RGRP inference."""

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt


    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 verifier=None,
                 search_schedule=[10, 15, 20],
                 candidates_size_schedule=[10, 5, 3, 1],
        ):
        """
        Generates video frames from text prompt using diffusion process.
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        search_iteration = 0
        number_of_N = candidates_size_schedule[search_iteration]

        noise_candidates= torch.randn(
                number_of_N,
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)

        STEP_TO_TIMESTEP =  {
                10: 200,
                15: 300,
                20: 400,
                25: 500,
                30: 600
            }

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(number_of_N):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(number_of_N):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            latents_total = noise_candidates
            cumulative_scores = torch.zeros(len(latents_total), device=self.device)

            for j, t in enumerate(tqdm(timesteps)):

                latents_list = []

                if j in search_schedule:

                    search_iteration += 1
                    number_of_N = candidates_size_schedule[search_iteration]

                    # Evaluate all de-noised candidates
                    verifier_tensors = [latents_.permute(1, 0, 2, 3).contiguous() for latents_ in latents_total]
                    verifier_prompt = [input_prompt] * len(latents_total)
                    verifier_steps = torch.full((len(latents_total),), STEP_TO_TIMESTEP[j], device=self.device, dtype=torch.long)
                    verifier_inputs = verifier.prepare_batch(
                            videos=verifier_tensors,
                            prompts=verifier_prompt,
                            denoising_steps=verifier_steps,
                            num_frames=20,
                    )
                    step_reward_score = verifier(verifier_inputs).sum(dim=1)
                    # [1.7080, 1.4160, 1.5635, 1.7266, 1.5850, 1.3164, 1.2261, 1.3525, 1.4697, 1.4980]
                    step_scores_norm = (step_reward_score - step_reward_score.min()) / (step_reward_score.max() - step_reward_score.min() + 1e-8)
                    cumulative_scores[:len(step_scores_norm)] += step_scores_norm

                    _, topk_indices = torch.topk(
                        cumulative_scores[: len(step_reward_score)], k=number_of_N
                    )
                    # topk_scores, topk_indices = torch.topk(step_reward_score, k=number_of_N)
                    # topk_scores: [1.7266, 1.7080, 1.5850, 1.5635, 1.4980]  topk_indices:  [3, 0, 4, 2, 9]

                    selected_latents = latents_total[topk_indices]
                    cumulative_scores = cumulative_scores[topk_indices]

                    sample_schedulers = []
                    for _ in range(number_of_N):
                        sample_scheduler = FlowDPMSolverMultistepScheduler(
                            num_train_timesteps=self.num_train_timesteps,
                            shift=1,
                            algorithm_type='sde-dpmsolver++',
                            use_dynamic_shifting=False)
                        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                        timesteps, _ = retrieve_timesteps(
                            sample_scheduler,
                            device=self.device,
                            sigmas=sampling_sigmas)
                        sample_schedulers.append(sample_scheduler)

                    latents_total = selected_latents

                # print(f"De-noising Step: {j+1}, Number of N: {number_of_N}")

                for k in range(number_of_N):

                    latents = latents_total[k]
                    latent_model_input = latents[None]

                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    latents_, _, _, _ = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)

                    latents_list.append(latents_)

                latents_total=torch.cat(latents_list)

            x0 = latents_total

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise_candidates, latents_total
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


    def generate_saving_all_videos(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='dpm++',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 verifier=None,
                 search_schedule=[10, 15, 20],
                 candidates_size_schedule=[6, 6, 6, 6],
                 beta=0,
        ):
        """
        Generates video frames from text prompt using diffusion process.
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        search_iteration = 0
        number_of_N = candidates_size_schedule[search_iteration]
        num_variants = number_of_N - 1

        noise_parent = torch.randn(
            1,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g
        )
        epsilons = torch.randn(
            num_variants,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g
        )
        noise_parent_expanded = noise_parent.expand(num_variants, -1, -1, -1, -1)
        scale = torch.sqrt(torch.tensor(1 - beta ** 2, device=noise_parent.device, dtype=noise_parent.dtype))
        noise_variants = scale * noise_parent_expanded + beta * epsilons
        noise_candidates = torch.cat([noise_parent, noise_variants], dim=0)

        STEP_TO_TIMESTEP =  {
                10: 200,
                15: 300,
                20: 400,
                25: 500,
                30: 600
            }

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(number_of_N):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(number_of_N):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            latents_total = noise_candidates
            cumulative_scores = torch.zeros(len(latents_total), device=self.device)
            step_reward_score_list = []
            step_scores_norm_list = []

            for j, t in enumerate(tqdm(timesteps)):

                latents_list = []

                if j in search_schedule:

                    search_iteration += 1
                    number_of_N = candidates_size_schedule[search_iteration]
                    _, _, n_frames, _, _  = latents_total.shape

                    # Evaluate all de-noised candidates and select top-k candidates
                    verifier_tensors = [latents_.permute(1, 0, 2, 3).contiguous() for latents_ in latents_total]
                    verifier_prompt = [input_prompt] * len(latents_total)
                    verifier_steps = torch.full((len(latents_total),), STEP_TO_TIMESTEP[j], device=self.device, dtype=torch.long)
                    verifier_inputs = verifier.prepare_batch(
                            videos=verifier_tensors,
                            prompts=verifier_prompt,
                            denoising_steps=verifier_steps,
                            num_frames=n_frames,
                    )
                    step_reward_score = verifier(verifier_inputs).sum(dim=1)
                    step_reward_score_list.append(step_reward_score)

                    step_scores_norm = FF.softmax(step_reward_score, dim=0)
                    step_scores_norm_list.append(step_scores_norm)

                    cumulative_scores[:len(step_scores_norm)] += step_scores_norm
                    topk_scores, topk_indices = cumulative_scores.sort(stable=True, descending=True)

                    topk_indices_list = topk_indices.tolist()
                    sample_schedulers = [sample_schedulers[i] for i in topk_indices_list]
                    selected_latents = latents_total[topk_indices]
                    cumulative_scores = cumulative_scores[topk_indices]

                    latents_total = selected_latents

                for k in range(number_of_N):

                    latents = latents_total[k]
                    latent_model_input = latents[None]

                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    latents_, _ = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)
                    # latents: torch.Size([1, 16, 21, 60, 104])
                    # noise_pred: torch.Size([16, 21, 60, 104])

                    latents_list.append(latents_)

                latents_total=torch.cat(latents_list)
                # latents_total: torch.Size([number_of_N, 16, 21, 60, 104])

            x0 = latents_total

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise_candidates, latents_total
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos if self.rank == 0 else None


    def generate_one_video(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 verifier=None,
                 search_schedule=[10, 15, 20],
                 candidates_size_schedule=[6, 4, 2, 1],
                 beta=0,
        ):
        """
        Generates video frames from text prompt using diffusion process.
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        search_iteration = 0
        number_of_N = candidates_size_schedule[search_iteration]
        num_variants = number_of_N - 1

        noise_parent = torch.randn(
            1,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g
        )
        epsilons = torch.randn(
            num_variants,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g
        )
        noise_parent_expanded = noise_parent.expand(num_variants, -1, -1, -1, -1)
        scale = torch.sqrt(torch.tensor(1 - beta ** 2, device=noise_parent.device, dtype=noise_parent.dtype))
        noise_variants = scale * noise_parent_expanded + beta * epsilons
        noise_candidates = torch.cat([noise_parent, noise_variants], dim=0)

        STEP_TO_TIMESTEP =  {
                10: 200,
                15: 300,
                20: 400,
                25: 500,
                30: 600
            }

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(number_of_N):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(number_of_N):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            latents_total = noise_candidates
            cumulative_scores = torch.zeros(len(latents_total), device=self.device)
            step_reward_score_list = []
            step_scores_norm_list = []

            for j, t in enumerate(tqdm(timesteps)):

                latents_list = []

                if j in search_schedule:

                    search_iteration += 1
                    number_of_N = candidates_size_schedule[search_iteration]
                    _, _, n_frames, _, _  = latents_total.shape

                    # Evaluate all de-noised candidates and select top-k candidates
                    verifier_tensors = [latents_.permute(1, 0, 2, 3).contiguous() for latents_ in latents_total]
                    verifier_prompt = [input_prompt] * len(latents_total)
                    verifier_steps = torch.full((len(latents_total),), STEP_TO_TIMESTEP[j], device=self.device, dtype=torch.long)
                    verifier_inputs = verifier.prepare_batch(
                            videos=verifier_tensors,
                            prompts=verifier_prompt,
                            denoising_steps=verifier_steps,
                            num_frames=n_frames,
                    )
                    step_reward_score = verifier(verifier_inputs).sum(dim=1)
                    step_scores_norm = FF.softmax(step_reward_score, dim=0)

                    cumulative_scores[:len(step_scores_norm)] += step_scores_norm
                    topk_scores, topk_indices = cumulative_scores.sort(stable=True, descending=True)
                    topk_indices = topk_indices[:number_of_N]

                    topk_indices_list = topk_indices.tolist()
                    sample_schedulers = [sample_schedulers[i] for i in topk_indices_list]
                    selected_latents = latents_total[topk_indices]
                    cumulative_scores = cumulative_scores[topk_indices]

                    latents_total = selected_latents

                for k in range(number_of_N):

                    latents = latents_total[k]
                    latent_model_input = latents[None]

                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    latents_, _ = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)

                    latents_list.append(latents_)

                latents_total=torch.cat(latents_list)

            x0 = latents_total

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise_candidates, latents_total
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


    def generate_with_latsearch(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 verifier=None,
                 search_schedule=None,
                 num_particles=6,
                 beta=0.1,
                 temperature=1.0,
        ):
        """Generate one video with LatSearch reward-guided resampling and pruning."""
        if search_schedule is None:
            search_schedule = [10, 15, 20]
        if not search_schedule:
            raise ValueError("search_schedule cannot be empty")
        if search_schedule != sorted(set(search_schedule)):
            raise ValueError("search_schedule must contain unique, increasing step indices")
        supported_search_steps = {10, 15, 20, 25, 30}
        if not set(search_schedule).issubset(supported_search_steps):
            raise ValueError(f"search_schedule must use steps in {sorted(supported_search_steps)}")
        if num_particles < 2:
            raise ValueError("num_particles must be at least 2")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        STEP_TO_TIMESTEP =  {
                10: 200,
                15: 300,
                20: 400,
                25: 500,
                30: 600
            }

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            noise_parent = torch.randn(
                1,
                *target_shape,
                dtype=torch.float32,
                device=self.device,
                generator=seed_g
            )
            epsilons = torch.randn(
                num_particles - 1,
                *target_shape,
                dtype=torch.float32,
                device=self.device,
                generator=seed_g
            )
            correlation = torch.sqrt(
                torch.tensor(1 - beta**2, device=noise_parent.device, dtype=noise_parent.dtype)
            )
            noise_variants = correlation * noise_parent + beta * epsilons
            latents_total = torch.cat([noise_parent, noise_variants], dim=0)  # [N, C, T, H, W]
            weights = torch.ones(num_particles, device=self.device) / num_particles
            cumulative_weights = torch.zeros(num_particles, device=self.device)

            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(num_particles):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(num_particles):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for j, t in enumerate(tqdm(timesteps)):

                latents_list = []

                for k in range(num_particles):

                    latents = latents_total[k]
                    latent_model_input = latents[None]

                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    latents_, _ = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)
                    latents_list.append(latents_)
                latents_total=torch.cat(latents_list)


                if j in search_schedule:

                    n_frames = latents_total.shape[2]
                    verifier_tensors = [l.permute(1, 0, 2, 3).contiguous() for l in latents_total]
                    verifier_prompt = [input_prompt] * num_particles
                    verifier_steps = torch.full((num_particles,), STEP_TO_TIMESTEP[j], device=self.device, dtype=torch.long)

                    verifier_inputs = verifier.prepare_batch(
                        videos=verifier_tensors,
                        prompts=verifier_prompt,
                        denoising_steps=verifier_steps,
                        num_frames=n_frames,
                    )
                    reward_scores = verifier(verifier_inputs).sum(dim=1)

                    # temperature scaling
                    logits = reward_scores / temperature
                    log_weights = logits - logits.max()
                    weights = torch.softmax(log_weights, dim=0)

                    cumulative_weights += weights

                    resample_indices = torch.multinomial(weights, num_particles, replacement=True)
                    resample_indices = torch.unique(resample_indices)

                    latents_total = latents_total[resample_indices]
                    cumulative_weights = cumulative_weights[resample_indices]
                    sample_schedulers = [sample_schedulers[i] for i in resample_indices.tolist()]
                    num_particles = len(resample_indices)

                    if j == search_schedule[-1]:
                        top_key = torch.argmax(cumulative_weights)
                        latents_total = latents_total[top_key].unsqueeze(0)
                        sample_schedulers = [sample_schedulers[top_key.item()]]
                        num_particles = 1

            x0 = latents_total

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del latents_total
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def generate_one_video_with_SMC(
            self,
            input_prompt,
            size=(1280, 720),
            frame_num=81,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            verifier=None,
            search_schedule=None,
            num_particles=6,
            beta=0.1,
            temperature=1.0,
        ):
        """Compatibility alias for :meth:`generate_with_latsearch`."""
        return self.generate_with_latsearch(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
            verifier=verifier,
            search_schedule=search_schedule,
            num_particles=num_particles,
            beta=beta,
            temperature=temperature,
        )


    def generate_one_video_with_SMC_cope(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 verifier=None,
                 search_schedule=[10, 15, 20],
                 num_particles=6,
                 beta=0.1,
        ):
        """
        Generates video frames from text prompt using diffusion process.
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        STEP_TO_TIMESTEP =  {
                10: 200,
                15: 300,
                20: 400,
                25: 500,
                30: 600
            }

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            noise_parent = torch.randn(
                1,
                *target_shape,
                dtype=torch.float32,
                device=self.device,
                generator=seed_g
            )
            epsilons = torch.randn(
                num_particles - 1,
                *target_shape,
                dtype=torch.float32,
                device=self.device,
                generator=seed_g
            )
            noise_variants = torch.sqrt(torch.tensor(1 - beta ** 2)) * noise_parent + beta * epsilons
            latents_total = torch.cat([noise_parent, noise_variants], dim=0)  # [N, C, T, H, W]
            weights = torch.ones(num_particles, device=self.device) / num_particles
            cumulative_weights = torch.zeros(num_particles, device=self.device)

            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(num_particles):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(num_particles):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for j, t in enumerate(tqdm(timesteps)):

                latents_list = []

                for k in range(num_particles):

                    latents = latents_total[k]
                    latent_model_input = latents[None]

                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    latents_, _, _, _ = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)
                    latents_list.append(latents_)
                latents_total=torch.cat(latents_list)

                if j in search_schedule:
                    n_frames = latents_total.shape[2]
                    verifier_tensors = [l.permute(1, 0, 2, 3).contiguous() for l in latents_total]
                    verifier_prompt = [input_prompt] * num_particles
                    verifier_steps = torch.full((num_particles,), STEP_TO_TIMESTEP[j], device=self.device, dtype=torch.long)

                    verifier_inputs = verifier.prepare_batch(
                        videos=verifier_tensors,
                        prompts=verifier_prompt,
                        denoising_steps=verifier_steps,
                        num_frames=n_frames,
                    )
                    reward_scores = verifier(verifier_inputs).sum(dim=1)
                    log_weights = reward_scores - reward_scores.max()
                    weights = torch.softmax(log_weights, dim=0)

                    cumulative_weights += weights

                    resample_indices = torch.multinomial(weights, num_particles, replacement=True)
                    resample_indices = torch.unique(resample_indices)

                    latents_total = latents_total[resample_indices]
                    cumulative_weights = cumulative_weights[resample_indices]
                    sample_schedulers = [sample_schedulers[i] for i in resample_indices.tolist()]
                    num_particles = len(resample_indices)

                    if j == search_schedule[-1]:
                        top_key = torch.argmax(cumulative_weights)
                        latents_total = latents_total[top_key].unsqueeze(0)
                        sample_schedulers = [sample_schedulers[top_key.item()]]
                        num_particles = 1

            x0 = latents_total

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del latents_total
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


# Backward compatibility for pre-release experiment scripts.
WanT2VWithLatentReward = WanT2VWithLatSearch


class WanT2VWithFreeInit:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        self.freq_filter = None

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
            self,
            input_prompt,
            size=(1280, 720),
            frame_num=81,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            num_iters: int = 5,
        ):

        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        self.freq_filter = get_freq_filter(
            [1, *target_shape],
            device=self.device,
            filter_type='butterworth',
            n=4,
            d_s=0.25,
            d_t=0.25,
        )

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        initial_noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(num_iters):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(num_iters):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for iter in range(num_iters):
                if iter == 0:
                    latents = initial_noise
                else:
                    latents_temp = latents[0].unsqueeze(0)
                    # 1. Forward with initial noise, get noisy latents z_T
                    current_diffuse_timestep = 999
                    diffuse_timesteps = torch.full((1,),int(current_diffuse_timestep))
                    diffuse_timesteps = diffuse_timesteps.long()
                    z_T = sample_schedulers[iter].add_noise(
                        original_samples=latents_temp.to(self.device),
                        noise=initial_noise[0].unsqueeze(0).to(self.device),
                        timesteps=diffuse_timesteps.to(self.device)
                    )
                    # 2. create random noise z_rand for high-frequency
                    z_rand = torch.randn(
                        1,
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=self.device,
                        generator=seed_g
                    )
                    # 3. Noise Reinitialization
                    latents_temp = freq_mix_3d(z_T.to(dtype=torch.float32), z_rand, LPF=self.freq_filter)
                    latents_temp = latents_temp.to(torch.float32)
                    latents = [latents_temp.squeeze(0)]

                for _, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents
                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    temp_x0, _ = sample_schedulers[iter].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)
                    latents = [temp_x0.squeeze(0)]
                    # latents: torch.Size([1, 16, 21, 60, 104])

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del latents
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


class WanT2VWithFreqPrior:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        self.freq_filter = None

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
            self,
            input_prompt,
            size=(1280, 720),
            frame_num=81,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            num_iters: int = 3,
            ratio: float = 0.8,
        ):

        F = frame_num
        target_shape = (self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        self.freq_filter = get_freq_filter(
            [1, *target_shape],
            device=self.device,
            filter_type='butterworth',
            n=4,
            d_s=0.25,
            d_t=0.25,
        )

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        initial_noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(device_type="cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():

            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(num_iters):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                    timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(num_iters):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            mid_step = 768 # 30th step

            for iter in range(num_iters):
                if iter == 0:
                    latents = initial_noise
                else:
                    latents_temp = latents[0].unsqueeze(0)
                    # 1. Forward with initial noise, get noisy latents z_T
                    current_diffuse_timestep = self.num_train_timesteps - 1
                    scheduler = sample_schedulers[iter]
                    sigmas = scheduler.sigmas.to(latents_temp.device, latents_temp.dtype)
                    sigma1 = sigmas[scheduler.index_for_timestep(mid_step, scheduler.timesteps)]
                    sigma2 = sigmas[scheduler.index_for_timestep(current_diffuse_timestep, scheduler.timesteps)]
                    alpha1, sigma1_t = scheduler._sigma_to_alpha_sigma_t(sigma1)
                    alpha2, sigma2_t = scheduler._sigma_to_alpha_sigma_t(sigma2)
                    s = alpha2 / alpha1
                    z_T = s * latents_temp + ( (sigma2_t**2 - (s * sigma1_t)**2).sqrt() ) * initial_noise[0].unsqueeze(0)
                    z_T = z_T.to(dtype=torch.float32)

                    # 2. filtering
                    eta1, eta2 = np.random.normal(size=z_T.shape), np.random.normal(size=z_T.shape)
                    eta1 = torch.tensor(eta1, dtype=torch.float32, device=self.device)
                    eta2 = torch.tensor(eta2, dtype=torch.float32, device=self.device)
                    x1 = ratio * z_T + (1 - ratio ** 2) ** 0.5 * eta1
                    x2 = ratio * z_T + (1 - ratio ** 2) ** 0.5 * eta2
                    x1 = x1 / ((1 + ratio ** 2) ** 0.5)
                    x2 = x2 / ((1 + ratio ** 2) ** 0.5)

                    y1, y2 = torch.randn_like(z_T), torch.randn_like(z_T)
                    part1 = freq_mix_noise_3d(x1, y1, LPF=self.freq_filter, output_type="+")
                    part2 = freq_mix_noise_3d(x2, y2, LPF=self.freq_filter, output_type="-")
                    latents_temp = (part1 + part2) / (2 ** 0.5)
                    latents = [latents_temp.squeeze(0)]

                for _, t in enumerate(tqdm(timesteps)):
                    if iter != (num_iters-1) and t.item() == int(mid_step):
                        break
                    latent_model_input = latents
                    timestep = [t]
                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    temp_x0, _ = sample_schedulers[iter].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)
                    latents = [temp_x0.squeeze(0)]
                    # latents: torch.Size([1, 16, 21, 60, 104])

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del latents
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
