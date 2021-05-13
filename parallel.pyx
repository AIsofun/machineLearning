#distutils: language=c++

import numpy as np
cimport numpy as cnp
import cython
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cdef parallel_v1(x, column):
    for row in range(x.shape[0]):
        x[row, column] = column


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void parallel_v2(double[:,:] x, const long nrow, const long column) nooil :
    cdef long long row
    for row from 0 <= row < nrow by 1:
        x[row, column] = colum

cpdef parallel_v2_test():
    natrix = np.random.randn(100000, 10)
    cdef int i
    cdef long nrow = matrix.shape[0]
    cdef double[:,:] arg = np.asfortranarray(matrix, dtype=np.float64)
    for i in prange(10, nogil = True):
        parallel_v2(arg, nrow, i)
