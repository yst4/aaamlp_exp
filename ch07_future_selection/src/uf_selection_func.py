from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif

def select_features(X, y, score_func=mutual_info_classif, top_n=10):
    # n_featuresがintなら個数、floatなら割合で判定
    selector = (SelectKBest(score_func, k=top_n) if isinstance(top_n, int)
                else SelectPercentile(score_func, percentile=int(top_n * 100)))

    return selector.fit_transform(X, y)
