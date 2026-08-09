"""Configuration used by latent reward preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    meta_data: str = ""
    data_dir: str = ""
    meta_data_test: str = ""
    max_frame_pixels: int = 240 * 320
    num_frames: int | None = None
    fps: float = 2.0
    p_shuffle_frames: float = 0.0
    p_color_jitter: float = 0.0
    eval_dim: str | list[str] = "VQ"
    prompt_template_type: str = "none"
    add_noise: bool = False
    sample_type: str = "uniform"
    use_tied_data: bool = True
