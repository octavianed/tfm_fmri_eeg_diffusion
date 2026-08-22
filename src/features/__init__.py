"""Precompute and load frozen-model features (CLIP embeddings, VAE latents, PCA)."""
from .clip_model import (ClipBundle, encode_image_paths, encode_image_tensor,
                         encode_pil_images, load_clip)
from .precompute_clip_embeddings import ImagePathDataset, precompute_clip
from .precompute_vae_latents import (load_vae, load_vae_meta,
                                     precompute_vae_latents)
from .fit_vae_pca import fit_vae_pca
from .load_features import (clip_norm_reference, explained_variance,
                            inverse_pca_to_latent, load_pca_bundle,
                            load_split_features)
from .text_embeddings import (TextEmbeddingCache, encode_prompts,
                              load_text_cache, load_text_encoder,
                              precompute_text_embeddings, text_cache_dir,
                              text_cache_hash)

__all__ = [
    "ClipBundle", "load_clip", "encode_image_tensor", "encode_pil_images",
    "encode_image_paths", "ImagePathDataset", "precompute_clip", "load_vae",
    "load_vae_meta", "precompute_vae_latents", "fit_vae_pca",
    "load_split_features", "load_pca_bundle", "inverse_pca_to_latent",
    "explained_variance", "clip_norm_reference",
    "precompute_text_embeddings", "load_text_cache", "load_text_encoder",
    "encode_prompts", "TextEmbeddingCache", "text_cache_dir", "text_cache_hash",
]
