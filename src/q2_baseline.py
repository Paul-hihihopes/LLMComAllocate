"""Q2 classical B1 fit and transfer diagnostics; exports fixed E for six fits."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from common import BASE, B
from scaling_models import metrics
import plotstyle


def fit_classical(N, D, loss):
    initial = [loss.min() * .7, 3., .3, 3., .3]
    def residual(p):
        return predict_classical(p, N, D) - loss
    sol = least_squares(residual, initial,
                        bounds=([0, 1e-6, 1e-4, 1e-6, 1e-4],
                                [loss.min()*1.2, 1e3, 5, 1e3, 5]), max_nfev=40000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x


def predict_classical(p, N, D):
    E, A, a, Bcoef, b = p
    return E + A*N**(-a) + Bcoef*D**(-b)


def main():
    out = Path(BASE) / "results/q2"
    out.mkdir(parents=True, exist_ok=True)
    files = {"B1_fit": "pythia_training_log_existing.csv",
             "Cerebras_transfer": "cerebras_training_log.csv",
             "cross_family_transfer": "scaling_baseline.csv",
             "published_transfer": "published_scaling_data.csv"}
    frames = {label: pd.read_csv(Path(B)/name).dropna(subset=["val_loss"])
              for label, name in files.items()}
    train = frames["B1_fit"]
    params = fit_classical(*(train[k].to_numpy(float) for k in
                             ["N_params_B", "D_tokens_B", "val_loss"]))
    rows, predictions = [], []
    for label, frame in frames.items():
        y = frame.val_loss.to_numpy(float)
        pred = predict_classical(params, frame.N_params_B.to_numpy(), frame.D_tokens_B.to_numpy())
        rows.append({"dataset": label, **metrics(y, pred),
                     "debiased_R2_diagnostic": metrics(y, pred + (y-pred).mean())["R2"]})
        predictions.append(frame.assign(dataset=label, predicted_loss=pred))
    pd.DataFrame(rows).to_csv(out/"classical_performance.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions).to_csv(out/"classical_predictions.csv", index=False, encoding="utf-8-sig")
    state = {"classical": {"params": dict(zip(["E", "A", "a", "B", "b"], map(float, params))),
                            "n": len(train), "r2_pythia": rows[0]["R2"]},
             "sources_sha256": {str((Path(B)/f).relative_to(BASE)): hashlib.sha256((Path(B)/f).read_bytes()).hexdigest()
                                for f in files.values()},
             "validation": rows,
             "note": "Debiasing uses target mean: diagnostic only, not independent validation."}
    (out/"classical_baseline.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5))
    y = train.val_loss.to_numpy()
    ax.scatter(y, predictions[0].predicted_loss, s=10)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    ax.set(xlabel="Observed loss (B1)", ylabel="Predicted loss", title="Classical scaling baseline")
    fig.tight_layout()
    fig.savefig(out/"00_classical_baseline.png", dpi=180)
    plt.close(fig)
    print(json.dumps(state["classical"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
