
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation metrics for perturbation-response prediction.

Metrics
-------
- mse, mae
- pcc_delta: global Pearson correlation across the full matrix
- global_pcc_delta: explicit alias for pcc_delta
- mean_perturbation_pcc: mean Pearson correlation across perturbations
- median_perturbation_pcc: median Pearson correlation across perturbations
- valid_pcc_fraction: fraction of perturbations with defined Pearson correlation
- deg_recovery_top20/50/100: mean Top-K absolute-response overlap
- deg_recovery_topK_valid_fraction: fraction of perturbations for which
  predicted Top-K recovery is defined

Important
---------
If a prediction row has identical absolute values for all genes, its
predicted Top-K ranking is undefined. Such rows are excluded from the
Top-K recovery average. This correctly handles the no_change baseline.
"""

import numpy as np


def _corr(a, b):
    """Pearson correlation of two flattened arrays.

    Returns NaN if fewer than two valid observations remain or either
    array has zero variance.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape.")

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return float("nan")

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def _row_correlations(y_true, y_pred):
    """Calculate one Pearson correlation per perturbation."""
    correlations = np.full(y_true.shape[0], np.nan, dtype=np.float64)

    for i in range(y_true.shape[0]):
        correlations[i] = _corr(y_true[i], y_pred[i])

    return correlations


def _topk_recovery(y_true, y_pred, k):
    """Calculate Top-K absolute-response overlap per perturbation.

    Rows with an undefined predicted ranking (all absolute predictions
    equal) are excluded instead of assigning arbitrary Top-K genes.
    """
    n_perturbations, n_genes = y_true.shape
    kk = min(int(k), n_genes)

    if kk <= 0:
        raise ValueError("k and the number of genes must be positive.")

    row_scores = []

    for i in range(n_perturbations):
        true_abs = np.abs(y_true[i])
        pred_abs = np.abs(y_pred[i])

        # A constant absolute prediction has no meaningful Top-K ranking.
        if np.allclose(
            pred_abs,
            pred_abs[0],
            rtol=1e-12,
            atol=1e-15,
        ):
            continue

        true_top = set(np.argpartition(true_abs, -kk)[-kk:].tolist())
        pred_top = set(np.argpartition(pred_abs, -kk)[-kk:].tolist())

        overlap = len(true_top & pred_top) / kk
        row_scores.append(overlap)

    valid_fraction = (
        len(row_scores) / n_perturbations
        if n_perturbations > 0
        else float("nan")
    )

    mean_recovery = (
        float(np.mean(row_scores))
        if row_scores
        else float("nan")
    )

    return mean_recovery, float(valid_fraction)


def response_metrics(y_true, y_pred, top_ks=(20, 50, 100)):
    """Compute regression, correlation, and Top-K recovery metrics.

    Parameters
    ----------
    y_true : array-like, shape (n_perturbations, n_genes)
        Observed perturbation-response matrix.
    y_pred : array-like, same shape as y_true
        Predicted perturbation-response matrix.
    top_ks : iterable of int
        Values of K for Top-K absolute-response recovery.

    Returns
    -------
    dict
        Metrics suitable for conversion to a pandas DataFrame.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must both be 2D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted arrays differ in shape.")

    if y_true.shape[0] == 0 or y_true.shape[1] == 0:
        raise ValueError("Response matrices must not be empty.")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or infinite values.")

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains NaN or infinite values.")

    err = y_pred - y_true

    # Keep the original global PCC field for compatibility.
    global_pcc = _corr(y_true, y_pred)

    # New: calculate PCC separately for each perturbation.
    row_pcc = _row_correlations(y_true, y_pred)
    valid_pcc = np.isfinite(row_pcc)

    out = {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "pcc_delta": global_pcc,
        "global_pcc_delta": global_pcc,
        "mean_perturbation_pcc": (
            float(np.mean(row_pcc[valid_pcc]))
            if valid_pcc.any()
            else float("nan")
        ),
        "median_perturbation_pcc": (
            float(np.median(row_pcc[valid_pcc]))
            if valid_pcc.any()
            else float("nan")
        ),
        "valid_pcc_fraction": float(np.mean(valid_pcc)),
        "valid_pcc_count": int(valid_pcc.sum()),
    }

    for k in top_ks:
        recovery, valid_fraction = _topk_recovery(
            y_true, y_pred, k
        )

        out[f"deg_recovery_top{k}"] = recovery
        out[f"deg_recovery_top{k}_valid_fraction"] = (
            valid_fraction
        )

    return out

