# How to use the script:
# uv run python src/np_sparse2.py --n_rows 1000 --n_cols 1000000

import numpy as np
from scipy import sparse

import argparse

def run(n_rows,n_cols):
    example = np.random.binomial(1, p=0.05, size=(n_rows,n_cols))

    print(f"Size of dense array: {example.nbytes}")

    sparse_example = sparse.csr_matrix(example)

    print(f"Full size of sparse array: {sparse_example.data.nbytes}")

    full_size = (
        sparse_example.data.nbytes +
        sparse_example.indptr.nbytes +
        sparse_example.indices.nbytes
    )

    print(f"Full size of sparse array: {full_size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_rows",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=100000
    )

    args = parser.parse_args()

    run(n_rows=args.n_rows, n_cols=args.n_cols)
