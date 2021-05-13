# coding = 'utf-8'
import numpy as np
import pandas as pd
import datetime
import tm

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



def main():
#    y = np.random.randint(2, size=(5000, 1))
#    x = np.random.randint(10, size=(5000, 1))
#    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
#    result_1 = target_mean_v1(data, 'y', 'x')
#    result_2 = target_mean_v2(data, 'y', 'x')

#    diff = np.linalg.norm(result_1 - result_2)
#    print(diff)
    np.random.seed(0)
    y = np.random.randint(2, size=(5000, 1))
    np.savetxt('y3.txt',y)


    np.random.seed(0)
    x = np.random.randint(10, size=(5000, 1))
    np.savetxt('x3.txt',x)

    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    #tow different way
    sum1 = 0.0
    sum2 = 0.0
    for i in range(1000):
        start1 = datetime.datetime.utcnow()
        result_1 = tm.target_mean_v3(data, 'y', 'x')
        end1 = datetime.datetime.utcnow()
        sum1 += (end1 - start1).total_seconds()
        print("V3 used: ", (end1 - start1).total_seconds(), " seconds")

        start2 = datetime.datetime.utcnow()
        result_2 = tm.target_mean_v4(data, 'y', 'x')
        end2 =datetime.datetime.utcnow()
        sum2 += (end2 - start2).total_seconds()
        print("I  used: ", (end2 - start2).total_seconds(), " seconds")
    print("total times:", sum1/sum2)

if __name__ == '__main__':
    main()


