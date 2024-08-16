import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
import time
from tqdm import trange

def find_replicant(real, fake):
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
    b = np.dot(fake, real.T) * 2
    distance_matrix = a - b
    return np.min(distance_matrix, axis=0)


def each_group(train, test, fake, n_train, n_test, theta_percentile= 5 , batchsize= 1000, performance_metric= "f1"):

    distance_real = np.zeros(n_train)
    distance_train = np.zeros(n_train)
    distance_test = np.zeros(n_test)

    steps = np.ceil(n_train / batchsize)

    for i in trange(int(steps), desc= "1/3 Calculating Real Distance"):
        mask = np.ones(len(train), bool)
        mask[i * batchsize:(i + 1) * batchsize] = False
        distance_real[i * batchsize:(i + 1) * batchsize] = find_replicant(train[i * batchsize:(i + 1) * batchsize], train[mask])

    for i in trange(int(steps), desc= "2/3 Calculating Train Distance"):
        distance_train[i * batchsize:(i + 1) * batchsize] = find_replicant(train[i * batchsize:(i + 1) * batchsize], fake)

    steps = np.ceil(n_test / batchsize)
    for i in trange(int(steps), desc= "3/3 Calculating Test Distance"):
        distance_test[i * batchsize:(i + 1) * batchsize] = find_replicant(test[i * batchsize:(i + 1) * batchsize], fake)

    theta = np.percentile(distance_real, theta_percentile)

    n_tp = np.sum(distance_train <= theta)  # true positive counts
    n_fn = n_train - n_tp
    n_fp = np.sum(distance_test <= theta)# false positive counts
    n_fp = int(n_fp * (n_train/n_test))
    n_tn = n_train - n_fp

    if performance_metric == "precision":
        precision = n_tp / (n_tp + n_fp)
        return precision
    elif performance_metric == "recall":
        sensitivity = n_tp / (n_tp + n_fn)
        return sensitivity
    else:
        f1 = n_tp / (n_tp + (n_fp + n_fn) / 2)  # F1 score
        return f1



def calculate_mir(train, test, fake, cont_cols, theta_percentile=10, batchsize=1000, performance_metric= "f1"):
    start1 = time.time()
    n_train = np.shape(train)[0]
    n_test = np.shape(test)[0]
    elapsed1 = (time.time() - start1)
    start2 = time.time()
    # normalization
    [n_row, n_col] = fake.shape

    norm = Normalizer().fit(fake[cont_cols])

    fake[cont_cols] = norm.transform(fake[cont_cols])
    train[cont_cols] = norm.transform(train[cont_cols])
    test[cont_cols] = norm.transform(test[cont_cols])

    result = np.abs(each_group(train.values, test.values, fake.values, n_train, n_test, theta_percentile=theta_percentile, batchsize=batchsize, performance_metric= performance_metric))
    elapsed2 = (time.time() - start2)
    print("Risk: " + str(result) + ".")
    print("Time used: " + str(elapsed1 + elapsed2) + " seconds.")
    print("Loading time used: " + str(elapsed1) + " seconds.")
    print("Computing time used: " + str(elapsed2) + " seconds.")
    return result