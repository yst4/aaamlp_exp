from sklearn import tree
from sklearn import ensemble
models = {
    "dicision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini",
    ),
    "dicision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier(),
}
