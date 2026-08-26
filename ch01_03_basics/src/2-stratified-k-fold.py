import pandas as pd
from sklearn import model_selection

if __name__== "__main__":
    df = pd.read_csv("train.csv")

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    # if you need reproducibility, rewrite the below, to fix the random seed:
    #   df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits=5)
    # if shuffling and fixed a random seed, please delete df = df.sample ... line.
    # kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for f, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = f

    df.to_csv("train_folds.csv",index=False)
