# coding = 'utf-8'
import numpy as np
import pandas as pd
import datetime
import tm

#v1 is the slowest way.
def target_mean_py_v1(data,y_name,x_name):
    result = np.zeros(0)
    groupby = data.groupby(x_name)
    sum_dict = groupby.agg(['sum'])[y_name].to_dict()['sum']
    count_dict = groupby.count()[y_name].to_dict()
    for y, x in zip(data[y_name], data[x_name]):
        result = np.append(result, ((sum_dict[x] - y) / (count_dict[x] - 1)))
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
    y = np.random.randint(2, size=(100000, 1))


    np.random.seed(0)
    x = np.random.randint(10, size=(100000, 1))


    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    # tow different way
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sum4 = 0.0
    for i in range(1):
        start1 = datetime.datetime.utcnow()
        result_1 = tm.target_mean_v3(data, 'y', 'x')
        end1 = datetime.datetime.utcnow()
        sum1 += (end1 - start1).total_seconds()
        print("V3 used: ", (end1 - start1).total_seconds(), " seconds")

        start2 = datetime.datetime.utcnow()
        result_2 = tm.target_mean_v4(data, 'y', 'x')
        end2 = datetime.datetime.utcnow()
        #sum2 += (end2 - start2).total_seconds()
        print("I  used: ", (end2 - start2).total_seconds(), " seconds")
    # print("total times:", sum1 / sum2)


    start3 = datetime.datetime.utcnow()
    result_3 = tm.target_mean_v5(data, 'y', 'x')
    end3 = datetime.datetime.utcnow()
    #sum3 += (end3 - start3).total_seconds()
    print("v5  used: ", (end3 - start3).total_seconds(), " seconds")


    start4 = datetime.datetime.utcnow()
    result_4 = target_mean_py_v1(data, 'y', 'x')
    end4 = datetime.datetime.utcnow()
    #sum4 += (end3 - start3).total_seconds()
    print("v1  used: ", (end4 - start4).total_seconds(), " seconds")


    start6 = datetime.datetime.utcnow()
    result_6 = tm.target_mean_v6(data, 'y', 'x')
    end6 = datetime.datetime.utcnow()
    #sum6 += (end6 - start6).total_seconds()
    print("v6 used: ", (end6 - start6).total_seconds(), " seconds")

    start7 = datetime.datetime.utcnow()
    result_7 = tm.target_mean_v7_test_pymp(data, 'y', 'x')
    end7 = datetime.datetime.utcnow()
    #sum7 += (end7 - start7).total_seconds()
    print("v7 used: ", (end7 - start7).total_seconds(), " seconds")

if __name__ == '__main__':
    main()
