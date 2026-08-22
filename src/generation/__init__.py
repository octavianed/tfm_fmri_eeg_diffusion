"""Generation with a frozen Stable Diffusion pipeline (spec §10, §11)."""
from .conditioning import (ARCHITECTURES, ConditionSpec,
                           assert_adapter_compatible, concat_condition,
                           condition_layout, conditioning_metadata,
                           controlnet_settings, describe_conditions,
                           num_neural_tokens,
                           required_brain_conditions, resolve_architecture,
                           resolve_conditions, uses_controlnet, uses_text)
from .controlnet_condition import (build_control_images, canny_edges,
                                   control_images_from_lowlevel,
                                   precompute_controlnet_conditions)
from .sd_pipeline import (FrozenSDGenerator, load_controlnet, load_sd_pipeline,
                          train_token_adapter)
from .generate_from_fmri import (build_control_inputs, build_text_inputs,
                                 generate_images, load_decoder,
                                 lowlevel_init_images,
                                 predict_condition_embeddings,
                                 resolve_adapter_checkpoint,
                                 resolve_clip_rescale,
                                 save_condition_images, select_samples)
from .make_grids import case_grids, comparison_grid, save_comparison_grid
from .checkpoint_sweep import (discover_adapter_checkpoints, margin_table,
                               save_sweep_figure, sweep_adapter_checkpoints)
from .input_scale_sweep import (save_input_scale_figure,
                                sweep_adapter_input_scale)
from .controlnet_scale_sweep import (save_controlnet_sweep_figure,
                                     sweep_controlnet_scale)

__all__ = [
    "FrozenSDGenerator", "load_sd_pipeline", "load_controlnet",
    "train_token_adapter",
    "generate_images", "load_decoder", "select_samples",
    "predict_condition_embeddings", "lowlevel_init_images",
    "resolve_clip_rescale", "resolve_adapter_checkpoint",
    "build_text_inputs", "build_control_inputs",
    "ARCHITECTURES", "ConditionSpec", "resolve_architecture",
    "resolve_conditions", "required_brain_conditions", "condition_layout",
    "conditioning_metadata", "controlnet_settings", "num_neural_tokens",
    "describe_conditions",
    "uses_text", "uses_controlnet", "concat_condition",
    "assert_adapter_compatible", "canny_edges", "build_control_images",
    "control_images_from_lowlevel", "precompute_controlnet_conditions",
    "save_condition_images", "comparison_grid", "save_comparison_grid",
    "case_grids", "discover_adapter_checkpoints", "sweep_adapter_checkpoints",
    "save_sweep_figure", "margin_table", "sweep_adapter_input_scale",
    "save_input_scale_figure", "sweep_controlnet_scale",
    "save_controlnet_sweep_figure",
]
