import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from functools import partial
from itertools import takewhile

def evaluate_subset(X, y, estimator, feature_indices):
    """指定サブセットのAUCを算出（特徴量が空なら0）"""
    if not feature_indices:
        return 0.0
    return float(np.mean(cross_val_score(estimator, X[:, feature_indices], y, cv=3, scoring='roc_auc')))

def feature_step(X, y, estimator, state):
    """現在の選択状態から『次にベストな1点』を加えた新しい状態を返す"""
    selected, score = state
    remaining = set(range(X.shape[1])) - set(selected)

    if not remaining:
        return selected, score

    # 残り特徴量を試行し、最もスコアが高いものを抽出
    eval_fn = partial(evaluate_subset, X, y, estimator)
    best_feat, best_score = max(
        ((f, eval_fn(selected + [f])) for f in remaining),
        key=lambda item: item[1]
    )

    return (selected + [best_feat], best_score) if best_score > score else (selected, score)

def step_stream(initial_state, step_fn):
    """状態の更新過程を無限ストリーム（ジェネレータ）として出力"""
    current = initial_state
    while True:
        yield current
        nxt = step_fn(current)
        if nxt == current:  # 変化がなくなったらストリーム終了
            break
        current = nxt

def greedy_feature_selection(X, y, estimator=None):
    model = estimator or LogisticRegression()
    step = partial(feature_step, X, y, model)

    # イテレーションの履歴を取得（スコアが伸びなくなったら自動停止）
    history = list(step_stream(([], 0.0), step))
    final_features, final_score = history[-1]

    return X[:, final_features], history

if __name__ == "__main__":
    X, y = make_classification(n_samples=500, n_features=15, n_informative=5, random_state=42)
    X_selected, history = greedy_feature_selection(X, y)

    print("選択された特徴量:", history[-1][0])
    print("スコア推移:", [round(s, 4) for _, s in history if s > 0])
