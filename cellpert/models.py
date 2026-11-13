import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class ResponsePredictor:
    """Baselines trained on perturbation-level average response vectors."""
    def __init__(self, model_name="ridge_gene", alpha=10.0, random_state=42):
        self.model_name = model_name
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.model = None
        self.train_mean_ = None
        self.n_outputs_ = None

    def fit(self, X, Y):
        X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
        if Y.ndim != 2 or X.ndim != 2 or len(X) != len(Y):
            raise ValueError("X and Y must be 2D and have matching rows")
        self.n_outputs_ = Y.shape[1]
        self.train_mean_ = Y.mean(axis=0)
        if self.model_name in {"no_change", "train_mean"}:
            return self
        if self.model_name == "ridge_gene":
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        elif self.model_name == "mlp_gene":
            self.model = make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400,
                             early_stopping=False, random_state=self.random_state,
                             alpha=1e-3, learning_rate_init=1e-3))
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        X = np.asarray(X, np.float32)
        if self.n_outputs_ is None: raise RuntimeError("Model is not fitted")
        if self.model_name == "no_change":
            return np.zeros((len(X), self.n_outputs_), np.float32)
        if self.model_name == "train_mean":
            return np.repeat(self.train_mean_[None, :], len(X), axis=0)
        return np.asarray(self.model.predict(X), np.float32)
