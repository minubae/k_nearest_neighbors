import numpy as np
import pandas as pd
from itertools import combinations
from numpy.linalg import norm
from numpy import sum
from numpy import append
from numpy import delete
from numpy import sort
from numpy import array as vec
from numpy import argpartition
from numpy import argmax
from numpy import bincount

# define column names
names = ['species_no', 'sepcies_name', 'petal_width', 'petal_length', 'sepal_width', 'sepal_length']

# loading training data
# data = pd.read_csv('iris_test.csv', header=None, names=names)
data = pd.read_csv('iris.csv')
data = vec(data)
# print(data.shape)
# print(data)
# print(data[0:5,2:4])
# data_train = data[0:10,2:]
data = data[0:15,:]
n = len(data)
print(data)
print(n)
print('')
K = 3
k_fold = 5

miss_classification_vec = []
train_data = []
test_data = []
corr_count = []


for i in range(0, n, k_fold):
    print('k_fold starts:')
    print(i)
    # print(data[i:i+k_fold])
    train_data = delete(data, range(i, i+k_fold), 0)
    test_data = data[i:i+k_fold]
    print('test data:')
    print(test_data)
    print('train data:')
    print(train_data)



    print('')


    for i, test in enumerate(test_data):

        print('test:', i)
        print(test[0], test[2:])

        dist_vec = []
        knn_vec = []


        for j, train in enumerate(train_data):
            # print('train:', j)
            # print(train[0], train[2:])
            dist = norm(train[2:]-test[2:])
            dist_vec.append(dist)

        print(i, dist_vec)

        k_indeices_vec = sort(argpartition(dist_vec, K)[:K], axis=0)
        print(k_indeices_vec)
        print(train_data)

        for k in k_indeices_vec:

            knn_train = train_data[k]
            knn_vec.append(knn_train[0])
            print(knn_train)

        print(knn_vec)
        counts = bincount(knn_vec)
        majority_vote = argmax(counts)

        print('majority vote:')
        print(majority_vote)
        print('this test data, ', test)
        print('is classified as species number ', majority_vote,'.')
        # print(test[0])


        count = 0
        for knn in knn_vec:
            if knn == test[0]:
                count += 1

        corr_count.append(count)

        print('correct count: ', count)
        print(corr_count)



        print('')



    print('')


print(corr_count)
total_correct = sum(corr_count)
print(total_correct)
denom = (n/k_fold)*(K*k_fold)
print(denom)

miss_classification_error = 1 - (total_correct / denom)
print('error rate: ', miss_classification_error)

miss_classification_vec.append(miss_classification_error)
print(miss_classification_vec)
