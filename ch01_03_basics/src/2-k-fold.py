import pandas as pd
from sklearn import model_selection

if __name__== "__main__":
    df = pd.read_csv("train.csv")

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    # if you need reproducibility, rewrite the below, to fix the random seed:
    #   df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    df.to_csv("train_folds.csv",index=False)
