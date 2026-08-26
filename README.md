## Caution for reading AAAMLP

This book was published in 2020. There are many API changes compared to the latest versions of the libraries used in the book.

### バージョン差分でハマりやすい主なポイント

- **transformers (2.11.0 → 4.x/5.x)**
  APIの変更が最も激しいライブラリです。関数の引数名やモデルのロード方法、Trainer 周りの仕様が現在と大きく異なります。

- **torch (1.5.0 → 2.x)**
  CUDA（GPU）対応の記述や、PyTorch 2.0以降の `torch.compile` / 推論処理の最適化周りで書き換えが必要になります。

- **scikit-learn (0.22.1 → 1.x)**
  キーワード引数の強制化（位置引数の廃止）や、一部のモジュール・APIの非推奨化／削除が行われています。

- **pandas (1.0.4 → 2.x)**
  `append()` メソッドの廃止や、`inplace=True` の非推奨化など、日常的に使う記法で警告・エラーが出ます。

---

This repository is managed by `uv`. Please refer to `pyproject.toml` for dependencies.
The code in this repository is rewritten to adapt to the updated APIs of these libraries.

## Dataset and Reference

- 📊 **Dataset**: https://www.kaggle.com/datasets/abhishek/aaamlp
- 📖 **PDF / Code**: https://github.com/abhishekkrthakur/approachingalmost
- 📖 **日本語版 (サポートページ）**: https://book.mynavi.jp/supportsite/detail/9784839974985.html

## Project structure.

```
$ tree aaamlp_exp/ -L 3
aaamlp_exp/
├── README.md
├── ch01_03_basics
│   ├── pyproject.toml
│   └── src
│       ├── 1-mnist.py
│       ├── 2-cross-valid-max-depth.py
│       ├── 2-cross-validation-refactoring.py
│       └── 2-cross-validation.py
├── ch04_categorical
│   ├── pyproject.toml
│   └── src
├── ch05_images
│   ├── README.md
│   ├── pyproject.toml
│   └── src
│       └── ch05_images
├── input
│   ├── adult.csv
│   ├── adult_folds.csv
│   ├── cat_test.csv
│   ├── cat_train.csv
│   ├── cat_train_folds.csv
│   ├── imdb.csv
│   ├── imdb_folds.csv
│   ├── mnist_test.csv
│   ├── mnist_train.csv
│   ├── mnist_train_folds.csv
│   ├── mobile_train.csv
│   └── winequality-red.csv
├── main.py
├── models
├── modify_points
│   ├── pandas.md
│   └── readme.md
├── notebooks
├── output
├── pyproject.toml
├── src
└── uv.lock
```

I don't save the repository about input and so on. 

```
$ cat .gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv

# 環境変数・機密情報
.env
.env.local
.env.*.local

# .gitignore に追記
input/
models/
output/
```
## Download AAAMLP datasets by kaggle cli

To unpack the dataset to the input from Kaggle site. 

```
$ mkdir input && cd input/
$ kaggle datasets download -d abhishek/aaamlp --path ./my_data --unzip
```
