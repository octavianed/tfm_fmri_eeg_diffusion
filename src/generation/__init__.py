"""Generation with a frozen Stable Diffusion pipeline (spec §10, §11)."""
from .sd_pipeline import (FrozenSDGenerator, load_sd_pipeline,
                          train_token_adapter)
from .generate_from_fmri import (generate_images, load_decoder,
                                 lowlevel_init_images,
                                 predict_condition_embeddings,
                                 resolve_clip_rescale,
                                 save_condition_images, select_samples)
from .make_grids import case_grids, comparison_grid, save_comparison_grid
from .checkpoint_sweep import (discover_adapter_checkpoints, margin_table,
                               save_sweep_figure, sweep_adapter_checkpoints)
from .input_scale_sweep import (save_input_scale_figure,
                                sweep_adapter_input_scale)

__all__ = [
    "FrozenSDGenerator", "load_sd_pipeline", "train_token_adapter",
    "generate_images", "load_decoder", "select_samples",
    "predict_condition_embeddings", "lowlevel_init_images",
    "resolve_clip_rescale",
    "save_condition_images", "comparison_grid", "save_comparison_grid",
    "case_grids", "discover_adapter_checkpoints", "sweep_adapter_checkpoints",
    "save_sweep_figure", "margin_table", "sweep_adapter_input_scale",
    "save_input_scale_figure",
]
