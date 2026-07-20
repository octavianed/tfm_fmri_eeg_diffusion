"""Evaluation: retrieval, embedding regression, ablations, baselines, generation."""
from .retrieval_metrics import compute_retrieval_metrics, topk_candidates
from .embedding_metrics import (embedding_regression_metrics,
                                per_component_pearson)
from .eval_data import SubjectMatrices, load_subject_matrices
from .baselines import (MeanBaseline, RidgeRegression, evaluate_baselines)
from .ablation_eval import (conclusion_from_summary, evaluate_ablation,
                            make_condition_input, sattolo_derangement,
                            save_ablation_figures)
from .generation_metrics import (clip_pairwise_similarity,
                                 compute_generation_metrics, pixel_mse)

__all__ = [
    "compute_retrieval_metrics", "topk_candidates",
    "embedding_regression_metrics", "per_component_pearson",
    "SubjectMatrices", "load_subject_matrices", "MeanBaseline",
    "RidgeRegression", "evaluate_baselines", "evaluate_ablation",
    "conclusion_from_summary", "make_condition_input", "sattolo_derangement",
    "save_ablation_figures",
    "compute_generation_metrics", "clip_pairwise_similarity", "pixel_mse",
]
