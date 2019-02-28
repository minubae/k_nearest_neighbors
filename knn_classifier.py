###########################################################################################################################
# Title: K-Nearest Neighbors (KNN) Classifier
# Date: 02/28/2019, Thursday
# Author: Minwoo Bae (minwoo.bae@uconn.edu)
# Institute: The Department of Computer Science and Engineering, UCONN
###########################################################################################################################

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
from numpy import ceil
from numpy import int

def get_majority_vote(knns):

    knn = knns
    counts = bincount(knn)
    vote_result = argmax(counts)

    return vote_result

def get_correctness(species_vec, correct_species_id):

    knn_v = species_vec
    correct_id = correct_species_id

    count = 0
    for knn in knn_v:
        if knn == correct_id:
            count += 1

    return count

def get_knn_id(knn_indices, k_train_data):

    knn_id_vec = []
    k_indices_v = knn_indices
    k_train_d = k_train_data

    for k in k_indices_v:

        knn_train = k_train_d[k]
        print('knn species id: ', knn_train[0])
        knn_id_vec.append(knn_train[0])

    return knn_id_vec


def get_missclassification_error(k_num, k_fold, collected_data):

    train_data = []
    test_data = []
    corr_count = []

    kf = k_fold
    coll_data = collected_data
    n = len(coll_data)

    K = k_num

    for idx in range(0, n, kf):

        train_data = delete(coll_data, range(idx, idx+kf), 0)
        test_data = coll_data[idx:idx+kf]

        print('***k_fold starts:', idx)
        print('k-fold test data:')
        print(test_data)
        print('train data:')
        print(train_data)
        print('')

        for i, test in enumerate(test_data):

            dist_vec = []
            knn_vec = []

            print('test data:', i, test)

            for j, train in enumerate(train_data):

                # compute a distance between train and test datas:
                dist = norm(train[2:]-test[2:])
                dist_vec.append(dist)

            # Find k-nearest neighbors
            # argpartition() gives K-nearest neighbors' indices in train data
            k_indices_vec = sort(argpartition(dist_vec, K)[:K], axis=0)
            print('KNN indices in a train data: ', k_indices_vec)

            # Find KNN's species number
            knn_vec = get_knn_id(k_indices_vec, train_data)

            # Get a majority vote to classify the test data
            majority_vote = get_majority_vote(knn_vec)

            print('majority vote:', majority_vote)
            print('this test data, ', test)
            print('is classified as species number ', majority_vote,'.')

            num_corr = get_correctness(knn_vec, test[0])
            corr_count.append(num_corr)

            print('correct count: ', num_corr)
            print(corr_count)
            print('')

        print('')

    total_correct = sum(corr_count)
    denom = (n/kf)*(K*kf)
    miss_classification_error = 1 - (total_correct / denom)

    # print(total_correct)
    # print(denom)
    print('error rate: ', miss_classification_error)

    return miss_classification_error

def main(input1, input2, data_matrix):

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    data_mat =data_matrix
    k_input = input1
    k_fold_input = input2

    miss_classification_vec = []
    K_vec = []

    loop_num = 40

    for K in range(1, loop_num, 2):

        miss_error = get_missclassification_error(K, k_fold_input, data_mat)
        miss_classification_vec.append(miss_error)
        K_vec.append(K)

    print(K_vec)
    print(miss_classification_vec)

    fig, ax = plt.subplots()
    # ax.yaxis.set_major_formatter(formatter)

    x = np.arange(len(K_vec))
    plt.bar(x, miss_classification_vec)
    plt.xticks(x, K_vec)

    plt.title('k-fold Cross Validation (k-fold = 5)')

    plt.plot(miss_classification_vec, 'r')
    plt.xlabel('The number of Neighbors K')
    plt.ylabel('Missclassification Error')
    plt.show()






if __name__ == '__main__':

    # loading training data
    data = pd.read_csv('iris.csv')
    data = vec(data)
    # data = data[0:15,:]

    K_input = 5
    k_fold = 5

    main(K_input, k_fold, data)
