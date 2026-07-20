"""Build comparison grids: [real | correct | permuted | zero] per sample (spec §11.4)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


def _label_strip(width: int, text: str, height: int = 22) -> Image.Image:
    strip = Image.new("RGB", (width, height), (245, 245, 245))
    ImageDraw.Draw(strip).text((4, 4), text, fill=(20, 20, 20))
    return strip


def comparison_grid(columns: Dict[str, List[Image.Image]],
                    column_order: Sequence[str], cell: int = 224,
                    row_ids: Optional[Sequence[str]] = None) -> Image.Image:
    """columns maps label -> list of images (one per row). Returns a grid image."""
    n_cols = len(column_order)
    n_rows = max(len(columns[c]) for c in column_order)
    label_h = 24
    id_w = 90 if row_ids is not None else 0
    grid = Image.new("RGB", (id_w + n_cols * cell, label_h + n_rows * cell),
                     (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for j, label in enumerate(column_order):
        draw.text((id_w + j * cell + 4, 6), label, fill=(0, 0, 0))
    for i in range(n_rows):
        if row_ids is not None and i < len(row_ids):
            draw.text((4, label_h + i * cell + cell // 2), str(row_ids[i])[:12],
                      fill=(0, 0, 0))
        for j, label in enumerate(column_order):
            imgs = columns[label]
            if i < len(imgs):
                grid.paste(imgs[i].convert("RGB").resize((cell, cell)),
                           (id_w + j * cell, label_h + i * cell))
    return grid


def save_comparison_grid(outputs: dict, out_path,
                         column_order=("real", "correct", "permuted", "zero"),
                         cell: int = 224, max_rows: Optional[int] = None) -> str:
    cols = {c: outputs[c] for c in column_order if c in outputs}
    ids = outputs.get("image_ids")
    if max_rows is not None:
        cols = {c: v[:max_rows] for c, v in cols.items()}
        ids = ids[:max_rows] if ids else None
    grid = comparison_grid(cols, [c for c in column_order if c in cols], cell, ids)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return str(out_path)


def case_grids(outputs: dict, per_sample_scores: np.ndarray, out_dir,
               column_order=("real", "correct", "permuted", "zero"),
               k: int = 5, cell: int = 224) -> dict:
    """Save best/median/worst grids ranked by ``per_sample_scores`` (higher=better)."""
    out_dir = Path(out_dir)
    order = np.argsort(-np.asarray(per_sample_scores))
    n = len(order)
    picks = {
        "best_cases": order[:k],
        "median_cases": order[max(0, n // 2 - k // 2): n // 2 - k // 2 + k],
        "worst_cases": order[-k:][::-1],
    }
    saved = {}
    for name, idxs in picks.items():
        sub = {c: [outputs[c][i] for i in idxs] for c in column_order
               if c in outputs}
        sub["image_ids"] = [outputs["image_ids"][i] for i in idxs] \
            if "image_ids" in outputs else None
        grid = comparison_grid(sub, [c for c in column_order if c in sub], cell,
                               sub.get("image_ids"))
        path = out_dir / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(path)
        saved[name] = str(path)
    return saved
