from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile,
    chi2, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression
)

# 判別用の型エイリアス
ProblemType = Literal["classification", "regression"]

@dataclass(frozen=True)
class FeatureSelector:
    method: str
    n_features: int | float
    problem_type: ProblemType = "classification"

    def _get_scoring_fn(self) -> Callable:
        match (self.problem_type, self.method):
            case ("classification", "f_classif"): return f_classif
            case ("classification", "chi2"): return chi2
            case ("classification", "mutual_info"): return mutual_info_classif
            case ("regression", "f_regression"): return f_regression
            case ("regression", "mutual_info"): return mutual_info_regression
            case _: raise ValueError(f"Invalid scoring configuration: {self.problem_type}, {self.method}")

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        scoring_fn = self._get_scoring_fn()

        match self.n_features:
            case int(k):
                selector = SelectKBest(score_func=scoring_fn, k=k)
            case float(p) if 0.0 < p <= 1.0:
                selector = SelectPercentile(score_func=scoring_fn, percentile=int(p * 100))
            case _:
                raise ValueError(f"Invalid n_features: {self.n_features}")

        return selector.fit_transform(X, y)
