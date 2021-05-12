# coding = 'utf-8'
import numpy as np
import pandas as pd
import datetime

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

cpdef target_mean_v3(data, yname, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype = np.float64)
    cdef np.ndarry[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarry[double] x = np.asfortranarray(data[x_name], dtype=np.float64)
    
    target_mean_v3_impl(result, y, x, nrow)
    return result


cpdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x):
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
    i = 0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]] - 1)
            

def main():
#    y = np.random.randint(2, size=(5000, 1))
#    x = np.random.randint(10, size=(5000, 1))
#    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
#    result_1 = target_mean_v1(data, 'y', 'x')
#    result_2 = target_mean_v2(data, 'y', 'x')

#    diff = np.linalg.norm(result_1 - result_2)
#    print(diff)

    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    start = datetime.time()

    result_1 = target_mean_v1(data, 'y', 'x')

    end =datetime.time()
    print(end - start)


if __name__ == '__main__':
    main()


