"""Visualisation helpers — predicted vs reference charge scatter, error histograms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from powelleem.model import EEMModel
    from powelleem.types import Dataset, ParamSet


def scatter_predicted_vs_reference(
    model: EEMModel,
    params: ParamSet,
    dataset: Dataset,
    path: str | Path,
    *,
    per_element: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Per-element scatter of predicted vs reference charges.

    Useful for diagnosing which atom types fit well and which need more
    training data or richer typing.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required. Install with `pip install powelleem[viz]`."
        ) from exc

    q_pred_flat = np.concatenate(model.predict(params, dataset))
    q_ref_flat = np.concatenate([m.target_charges for m in dataset.molecules])
    type_flat = np.concatenate([m.atom_types for m in dataset.molecules])

    if per_element:
        unique_types = sorted(set(type_flat.tolist()))
        n_cols = 4
        n_rows = int(np.ceil(len(unique_types) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        for ax, t_idx in zip(axes.ravel(), unique_types, strict=False):
            mask = type_flat == t_idx
            label = params.atom_types[t_idx - 1]
            ax.scatter(q_ref_flat[mask], q_pred_flat[mask], s=4, alpha=0.4)
            mn = float(np.min([q_ref_flat[mask].min(), q_pred_flat[mask].min()]))
            mx = float(np.max([q_ref_flat[mask].max(), q_pred_flat[mask].max()]))
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=0.8)
            ax.set_title(f"{label} (n={int(mask.sum())})")
            ax.grid(True, alpha=0.3)
        for ax in axes.ravel()[len(unique_types):]:
            ax.axis("off")
        fig.supxlabel("reference charge")
        fig.supylabel("predicted charge")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(q_ref_flat, q_pred_flat, s=4, alpha=0.4)
        mn = float(min(q_ref_flat.min(), q_pred_flat.min()))
        mx = float(max(q_ref_flat.max(), q_pred_flat.max()))
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=0.8)
        ax.set_xlabel("reference")
        ax.set_ylabel("predicted")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)

    import matplotlib.pyplot as plt2
    plt2.close(fig)


def error_histogram(
    model: EEMModel,
    params: ParamSet,
    dataset: Dataset,
    path: str | Path,
    *,
    bins: int = 60,
) -> None:
    """Distribution of per-atom prediction errors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required.") from exc

    r = model.residuals_flat(params, dataset)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r, bins=bins)
    ax.axvline(0, color="r", linestyle="--", linewidth=0.8)
    ax.set_xlabel("q_pred − q_ref (e)")
    ax.set_ylabel("count")
    ax.set_title(f"RMSE = {float(np.sqrt((r * r).mean())):.4f} e")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
