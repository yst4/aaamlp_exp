import numpy as np
from scipy import sparse

example = np.array (
    [
        [0,0,1],
        [1,0,0],
        [1,0,1]
    ]
)

print(example.nbytes)
sparse_example = sparse.csr_matrix(example)

print(sparse_example.data.nbytes)

full_size = (
    sparse_example.data.nbytes +
    sparse_example.indptr.nbytes +
    sparse_example.indices.nbytes
)

print(f"Full size of sparse array: {full_size}")
