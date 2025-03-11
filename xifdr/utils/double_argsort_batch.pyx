# cython: language_level=3
import cython
import numpy as np

# noinspection PyUnresolvedReferences
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list double_argsort_batch(list a, list b):
    if a is None or b is None:
        return None
    return [
        np.argsort(
            [f'{a[i][j]};{b[i][j]}' for j in range(len(a[i]))]
        ) for i in range(len(a))
    ]
