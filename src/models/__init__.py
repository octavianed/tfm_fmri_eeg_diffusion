"""Model components: fMRI/EEG encoders, prediction heads, adapters, multitask decoder."""
from .fmri_encoder import FMRIEncoder, ResidualMLPBlock, VoxelDropout
from .eeg_encoder import (ChannelDropout, EEGEncoderTemporalConv,
                          ResidualTemporalBlock, TemporalAttentionPooling)
from .heads import CLIPHead, LowLevelHead, ProjectionHead
from .adapters import SubjectAdapters, TokenAdapter
from .multitask_decoder import (MultitaskDecoder, build_model,
                                build_model_from_checkpoint)

__all__ = [
    "FMRIEncoder", "ResidualMLPBlock", "VoxelDropout", "ProjectionHead",
    "EEGEncoderTemporalConv", "ChannelDropout", "ResidualTemporalBlock",
    "TemporalAttentionPooling",
    "CLIPHead", "LowLevelHead", "SubjectAdapters", "TokenAdapter",
    "MultitaskDecoder", "build_model", "build_model_from_checkpoint",
]
