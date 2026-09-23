import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

def evaluate_subset(X, y, feature_indices, estimator, cv=3):
    """指定された特徴量サブセットの交差検証スコアを計算する（純粋関数）"""
    if not feature_indices:
        return 0.0
    scores = cross_val_score(estimator, X[:, feature_indices], y, cv=cv, scoring='roc_auc')
    return float(np.mean(scores))

def select_next_best_feature(X, y, current_features, estimator):
    """まだ選択されていない特徴量の中から、最もスコアが上がる1つを探索する"""
    n_features = X.shape[1]
    remaining = [f for f in range(n_features) if f not in current_features]

    candidates = (
        (f, evaluate_subset(X, y, current_features + [f], estimator))
        for f in remaining
    )
    # 最もスコアの高い特徴量とスコアのペアを返す（候補がなければ None）
    return max(candidates, key=lambda x: x[1], default=(None, 0.0))

def greedy_feature_selection(X, y, estimator=None):
    """Greedy Feature Selection のメインループ"""
    model = estimator or LogisticRegression()
    selected_features = []
    best_score = 0.0
    history = []

    while True:
        next_feature, score = select_next_best_feature(X, y, selected_features, model)

        # スコアが更新されない、または追加できる特徴量がなければ終了
        if next_feature is None or score <= best_score:
            break

        selected_features.append(next_feature)
        best_score = score
        history.append((next_feature, score))

    return X[:, selected_features], history

if __name__ == "__main__":
    X, y = make_classification(n_samples=500, n_features=15, n_informative=5, random_state=42)

    X_selected, history = greedy_feature_selection(X, y)

    print("選択された特徴量インデックス:", [f for f, _ in history])
    print("スコア推移:", [round(s, 4) for _, s in history])
