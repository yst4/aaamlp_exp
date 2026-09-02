import numpy as np
from scipy import sparse

example = np.array(
    [
        [0,0,0,0,1,0],
        [0,1,0,0,0,0],
        [1,0,0,0,0,0]
    ]
)

print(f"Size of dense array: {example.nbytes}")

sparse_example = sparse.csc_matrix(example)

print(f"Size of sparse array: {sparse_example.data.nbytes}")

full_size = (
    sparse_example.data.nbytes +
    sparse_example.indptr.nbytes +
    sparse_example.indices.nbytes
)

print(f"Full size of sparse array: {full_size}")
