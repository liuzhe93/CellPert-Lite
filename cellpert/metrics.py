import numpy as np

def _corr(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def response_metrics(y_true, y_pred, top_ks=(20, 50, 100)):
    y_true = np.asarray(y_true, np.float64); y_pred = np.asarray(y_pred, np.float64)
    if y_true.shape != y_pred.shape: raise ValueError("true and predicted arrays differ in shape")
    err = y_pred - y_true
    out = {"mse": float(np.mean(err**2)), "mae": float(np.mean(np.abs(err))),
           "pcc_delta": _corr(y_true, y_pred)}
    for k in top_ks:
        kk = min(int(k), y_true.shape[1]); vals = []
        for i in range(len(y_true)):
            t = set(np.argsort(np.abs(y_true[i]))[-kk:])
            p = set(np.argsort(np.abs(y_pred[i]))[-kk:])
            vals.append(len(t & p) / max(1, len(t)))
        out[f"deg_recovery_top{k}"] = float(np.mean(vals)) if vals else float("nan")
    return out
