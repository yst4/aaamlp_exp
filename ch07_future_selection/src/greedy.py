import numpy as np
from sklearn import linear_model, metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

class GreedyFeatureSelection:
    def __init__(self, estimator=None, cv=3):
        self.estimator = estimator or linear_model.LogisticRegression()
        self.cv = cv

    def evaluate_score(self, X, y):
        # リークを防ぐため交差検証でスコアを算出
        scores = cross_val_score(self.estimator, X, y, cv=self.cv, scoring='roc_auc')
        return np.mean(scores)

    def feature_selection(self, X, y):
        good_features = []
        best_scores = []
        num_features = X.shape[1]

        while True:
            this_feature = None
            best_score = 0.0

            # まだ選ばれていない特徴量を1つずつ試す
            for feature in range(num_features):
                if feature in good_features:
                    continue

                selected_features = good_features + [feature]
                xtrain = X[:, selected_features]
                score = self.evaluate_score(xtrain, y)

                if score > best_score:
                    this_feature = feature
                    best_score = score

            # スコアが改善しなかった、または追加できる特徴量がない場合は終了
            if this_feature is None:
                break

            if len(best_scores) > 0 and best_score <= best_scores[-1]:
                break

            good_features.append(this_feature)
            best_scores.append(best_score)

        return best_scores, good_features

    def __call__(self, X, y):
        scores, features = self.feature_selection(X, y)
        return X[:, features], scores

if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)

    selector = GreedyFeatureSelection()
    X_transformed, scores = selector(X, y)

    print(f"選ばれた特徴量の数: {X_transformed.shape[1]}")
    print(f"推移スコア: {[round(s, 4) for s in scores]}")
