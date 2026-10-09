"""Interchangeable 17-domain mixture regressions, with unpenalized intercepts.

M1 is interpreted literally: one signed w_k shared by all target domains.
M1_independent is a diagnostic relaxation, not the formula in the image.
M2 has one b_kj per input/target pair and a common positive tau.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


FAMILIES = ("linear", "M1", "M1_independent", "M2")


@dataclass
class MixtureModel:
    family: str
    alpha: float
    tau: float
    coef_: np.ndarray
    intercept_: np.ndarray

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all() or np.any(X < 0):
            raise ValueError("Mixture proportions must be finite and nonnegative")
        if self.family == "linear":
            return X
        return np.log1p(X / (self.tau if self.family == "M2" else 1.0))

    def predict(self, X):
        return self.transform(X) @ self.coef_.T + self.intercept_

    def marginal_effects(self, X):
        """Unconstrained partial derivatives; feasible changes must sum to zero."""
        X = np.asarray(X, dtype=float)
        if self.family == "linear":
            return np.broadcast_to(self.coef_, (len(X), *self.coef_.shape))
        tau = self.tau if self.family == "M2" else 1.0
        return self.coef_[None, :, :] / (tau + X[:, None, :])


def fit_mixture_model(X, Y, family="linear", alpha=1.0, tau=1.0):
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family}; use {FAMILIES}")
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(tau) or tau <= 0:
        raise ValueError("alpha must be finite/nonnegative and tau finite/positive")
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if not np.isfinite(Y).all() or len(X) != len(Y):
        raise ValueError("Losses must be finite and aligned with mixtures")
    model = MixtureModel(family, float(alpha), float(tau), None, None)
    Z = model.transform(X)
    # Shared objective: sum over all (i,j) residuals + alpha*||beta||^2.
    # Target averaging divides the objective by n_targets, including the penalty.
    if family == "M1":
        ridge = Ridge(alpha=alpha / Y.shape[1], solver="svd").fit(Z, Y.mean(axis=1))
        model.coef_ = np.broadcast_to(ridge.coef_, (Y.shape[1], X.shape[1])).copy()
        model.intercept_ = Y.mean(axis=0) - Z.mean(axis=0) @ model.coef_.T
    else:
        ridge = Ridge(alpha=alpha, solver="svd").fit(Z, Y)
        model.coef_, model.intercept_ = ridge.coef_, ridge.intercept_
    return model
