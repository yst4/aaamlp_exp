# How to use the script:
# uv run python src/many_futures.py --futures 1000 --size 1000000
import argparse
import numpy as np
from sklearn import preprocessing


def run(futures, size):
    example = np.random.randint(futures, size=size)

    # Dense array
    # original code (spars= is abolition): ohe = preprocessing.OneHotEncoder(sparse=False)
    ohe = preprocessing.OneHotEncoder(sparse_output=False)
    ohe_example = ohe.fit_transform(example.reshape(-1, 1))
    print(f"Size of dense array: {ohe_example.nbytes}")

    # Sparse array
    ohe = preprocessing.OneHotEncoder(sparse_output=True)
    ohe_example = ohe.fit_transform(example.reshape(-1, 1))
    print(f"Size of sparse array: {ohe_example.data.nbytes}")

    full_size = (
        ohe_example.data.nbytes +
        ohe_example.indptr.nbytes +
        ohe_example.indices.nbytes
    )
    print(f"Full size of sparse array: {full_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--futures",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100000
    )

    args = parser.parse_args()

    run(futures=args.futures, size=args.size)
