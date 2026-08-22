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
from .generation_ablation import (DELTA_DEFINITIONS, build_report,
                                  compute_deltas, conclusion, paired_test,
                                  score_conditions)

__all__ = [
    "compute_retrieval_metrics", "topk_candidates",
    "embedding_regression_metrics", "per_component_pearson",
    "SubjectMatrices", "load_subject_matrices", "MeanBaseline",
    "RidgeRegression", "evaluate_baselines", "evaluate_ablation",
    "conclusion_from_summary", "make_condition_input", "sattolo_derangement",
    "save_ablation_figures",
    "compute_generation_metrics", "clip_pairwise_similarity", "pixel_mse",
    "score_conditions", "compute_deltas", "paired_test", "conclusion",
    "build_report", "DELTA_DEFINITIONS",
]
