# distutils: language=c++
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libcpp.unordered_map cimport unordered_map
#from libcpp.map cimport map #use unordered_map instead
from cython cimport parallel
import pymp

def hello():
    print("hello")

def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


cpdef target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

    target_mean_v3_impl(result, y, x, nrow)
    return result

cdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1

    i=0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)
        


# homework
cpdef target_mean_v4(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

    target_mean_v4_impl(result, y, x, nrow)
    return result

cdef void target_mean_v4_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef unordered_map[double, double] value_dict
    cdef unordered_map[double, double] count_dict
    cdef long i

    for i in range(nrow):
        if value_dict.find(x[i]) != value_dict.end():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1

    i=0
    for i in range(nrow):
        if 0 != count_dict[x[i]] - 1:

            result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)
        else:
            result[i] = 0


cpdef target_mean_v5(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[int] x = np.asfortranarray(data[x_name], dtype=np.int32)
    target_mean_v5_impl(result, y, x, nrow)
    return result

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void target_mean_v5_impl(double[:] result, double[:] y, int[:] x, const long nrow):
    cdef unordered_map[int, double] value_dict
    cdef unordered_map[int, double] count_dict
    cdef long i
    for i in range(nrow):
        if  value_dict.find(x[i]) == value_dict.end():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1.0
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1.0
    i=0
    for i in prange(nrow,nogil=True,schedule='static', chunksize=1):
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)

cpdef target_mean_v6(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[int] x = np.asfortranarray(data[x_name], dtype=np.int32)
    cdef np.ndarray[int] y = np.asfortranarray(data[y_name], dtype=np.int32)

    cdef unordered_map[int, int] value_map
    cdef unordered_map[int, int] count_map

    for i from 0 <= i < nrow by 1:
        if value_map.count(x[i]):
            value_map[x[i]] += y[i]
            count_map[x[i]] += 1
        else:
            value_map[x[i]] = y[i]
            count_map[x[i]] = 1

    for i from 0 <= i < nrow by 1:
        result[i] = (value_map[x[i]] - y[i]) / (count_map[x[i]] - 1)
    return result

cpdef target_mean_v7_test_pymp(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = pymp.shared.array((nrow,), dtype=np.float64)
    cdef np.ndarray[int] x = np.asfortranarray(data[x_name], dtype=np.int32)
    cdef np.ndarray[int] y = np.asfortranarray(data[y_name], dtype=np.int32)

    cdef unordered_map[int, int] value_map
    cdef unordered_map[int, int] count_map

    for i from 0 <= i < nrow by 1:
        if value_map.count(x[i]):
            value_map[x[i]] = value_map[x[i]] + y[i]
            count_map[x[i]] += 1
        else:
            value_map[x[i]] = y[i]
            count_map[x[i]] = 1

    with pymp.Parallel(2) as p:
        for i in p.range(0, nrow):
            result[i] = (value_map[x[i]] - y[i]) / (count_map[x[i]] - 1)

    return result
