import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
import os
import sys

import tensorflow as tf
import keras
# TensorFlow 経由ではなく keras から直にインポート
import keras
from keras import layers
from keras import optimizers
from keras.models import Model
from keras import callbacks
from keras import utils

def create_model(data, catcols):
    """
    This function returns a compiled keras model
    for entity embeddings
    """
    inputs = []
    outputs = []

    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values) / 2), 50))

        # 入力レイヤー
        inp = layers.Input(shape=(1,), name=f"input_{c}")

        # Embedding レイヤー
        out = layers.Embedding(
            input_dim=num_unique_values + 1,
            output_dim=embed_dim,
            name=c
        )(inp)

        out = layers.SpatialDropout1D(0.3)(out)
        out = layers.Reshape(target_shape=(embed_dim,))(out)

        inputs.append(inp)
        outputs.append(out)

    # カテゴリ変数の Embedding 出力を結合
    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    # 出力レイヤー
    y = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=y)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def run(fold):
    df = pd.read_csv("./input/cat_train_folds.csv")

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

   # fill all NaN values with NONE and convert to string
    for col in features:
        df[col] = df[col].fillna("NONE").astype(str)

    # encode all features with label encoder individually
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df[feat] = lbl_enc.fit_transform(df[feat].values)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = create_model(df, features)

    # 各特徴量を配列としてリスト化
    xtrain = [
        df_train[features].values[:, k].astype(np.int32) for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k].astype(np.int32) for k in range(len(features))
    ]

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    model.fit(
        xtrain,
        ytrain_cat,
        validation_data=(xvalid, yvalid_cat),
        verbose=1,
        batch_size=1024,
        epochs=3
    )

    valid_preds = model.predict(xvalid)[:, 1]
    print(f"Fold {fold} ROC AUC: {metrics.roc_auc_score(yvalid, valid_preds)}")

    # Keras 3 でのセッションクリア
    keras.backend.clear_session()


if __name__ == "__main__":
    for fold in range(5):
        run(fold)
