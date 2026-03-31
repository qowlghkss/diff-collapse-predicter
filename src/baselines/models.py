import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class RandomPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, strategy="uniform", random_state=None):
        self.strategy = strategy
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.strategy == "stratified":
            self.class_prior_ = np.bincount(y) / len(y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        n_samples = len(X)
        if self.strategy == "uniform":
            probs = self.rng.random((n_samples, 2))
            probs = probs / probs.sum(axis=1, keepdims=True)
        else: # stratified
            probs = self.rng.choice(
                [0, 1], size=(n_samples, 2), p=self.class_prior_
            ) # This is a simplification
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class MajorityPredictor(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.majority_class_ = np.argmax(np.bincount(y))
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        n_samples = len(X)
        probs = np.zeros((n_samples, 2))
        probs[:, self.majority_class_] = 1.0
        return probs

    def predict(self, X):
        check_is_fitted(self)
        return np.full(len(X), self.majority_class_)

class HeuristicPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=None):
        self.threshold = threshold

    def fit(self, X, y):
        # X is expected to be a 1D array of heuristic values (e.g., mean CI)
        self.classes_ = np.unique(y)
        if self.threshold is None:
            # Simple grid search for best threshold on training data
            thresholds = np.linspace(np.min(X), np.max(X), 100)
            best_f1 = -1
            best_t = thresholds[0]
            
            from sklearn.metrics import f1_score
            for t in thresholds:
                preds = (X >= t).astype(int)
                score = f1_score(y, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = t
            self.threshold = best_t
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        # Use a sigmoid-like distance from threshold as a proxy for probability
        # Or just 0/1 for simplicity in a baseline
        preds = (X >= self.threshold).astype(int)
        probs = np.zeros((len(X), 2))
        probs[np.arange(len(X)), preds] = 1.0
        return probs

    def predict(self, X):
        check_is_fitted(self)
        return (X >= self.threshold).astype(int)
