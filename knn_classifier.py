import numpy as np
import pandas as pd
from itertools import combinations
from numpy.linalg import norm
from numpy import sum
from numpy import append
from numpy import delete
from numpy import array as vec

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
k_fold = 3

for i in range(0, n, k_fold):
    print(i)
    print(data[i:i+k_fold])



'''
k = 5
counter = 0
n = len(data_train)

index_vec = []
dist_vec = []
knn_indices = []

k_sum = 10**12

for i in range(n):
    index_vec.append(i)

index_vec = vec(index_vec)

for i in range(n):

    index_temp = delete(index_vec, i)
    choosK_combinations = combinations(index_temp, k)

    for k_index, combs in enumerate(choosK_combinations):
        print(i, combs,':',k_index)
        dist_vec = []

        for j in combs:
            dist = norm(data_train[j]-data_train[i])
            dist_vec.append(dist)

        k_sum_temp = sum(dist_vec)

        if k_sum_temp < k_sum:
            k_sum = k_sum_temp

            knn_indices = append(i, combs)

        print(k_sum_temp)
        print(k_sum)
        print('')

print(knn_indices)

for index in knn_indices:

    print(index)
    print(data_train[index])
'''
