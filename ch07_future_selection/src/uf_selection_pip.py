from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),
    ('classification', RandomForestClassifier())
])
pipeline.fit(X, y)
