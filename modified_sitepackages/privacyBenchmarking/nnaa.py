import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
import time
from tqdm import tqdm

def get_entropy(column):
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk = hist1 / len(column)
    entropy = -np.sum(pk * np.log(pk))
    return entropy


def find_replicant(real, fake):
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
    b = np.dot(fake, real.T) * 2
    distance_matrix = a - b
    return np.min(distance_matrix, axis=0)

def find_replicant_self(real, fake):
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
    b = np.dot(fake, real.T) * 2
    distance_matrix = a - b
    n_col = np.shape(distance_matrix)[1]
    min_distance = np.zeros(n_col)
    for i in range(n_col):
        sorted_column = np.sort(distance_matrix[:, i])
        min_distance[i] = sorted_column[1]
    return min_distance

def each_group(train, test, fake, n_train, n_test, batchsize= 1000):
    steps = np.ceil(n_test / batchsize).astype(int)
    if n_train == n_test:
        n_draw = 1
    else:
        n_draw = np.ceil(n_train / n_test).astype(int)
    # training dataset
    distance_train_TS = np.zeros(n_test)
    distance_train_TT = np.zeros(n_test)
    distance_train_ST = np.zeros(n_test)
    distance_train_SS = np.zeros(n_test)
    aa_train = 0

    with tqdm(total=n_draw*steps, desc= "1/2 Compare: Train - Synth") as pbar:
        for ii in range(n_draw):
            np.random.seed(ii)
            train_sample = np.random.permutation(train)[:n_test]
            np.random.seed(ii)
            fake_sample = np.random.permutation(fake)[:n_test]
            for i in range(steps):
                distance_train_TS[i * batchsize:(i + 1) * batchsize] = find_replicant(train_sample[i * batchsize:(i + 1) * batchsize], fake_sample)
                distance_train_ST[i * batchsize:(i + 1) * batchsize] = find_replicant(fake_sample[i * batchsize:(i + 1) * batchsize], train_sample)
                distance_train_TT[i * batchsize:(i + 1) * batchsize] = find_replicant_self(train_sample[i * batchsize:(i + 1) * batchsize], train_sample)
                distance_train_SS[i * batchsize:(i + 1) * batchsize] = find_replicant_self(fake_sample[i * batchsize:(i + 1) * batchsize], fake_sample)
                pbar.update(1)
            aa_train += (np.sum(distance_train_TS > distance_train_TT) + np.sum(distance_train_ST > distance_train_SS)) / n_test / 2
        aa_train /= n_draw

    # test dataset
    distance_test_TS = np.zeros(n_test)
    distance_test_TT = np.zeros(n_test)
    distance_test_ST = np.zeros(n_test)
    distance_test_SS = np.zeros(n_test)
    aa_test = 0
    with tqdm(total=n_draw * steps, desc="2/2 Compare: Test - Synth") as pbar:
        for ii in range(n_draw):
            np.random.seed(ii)
            fake_sample = np.random.permutation(fake)[:n_test]
            for i in range(steps):
                distance_test_TS[i * batchsize:(i + 1) * batchsize] = find_replicant(test[i * batchsize:(i + 1) * batchsize], fake_sample)
                distance_test_ST[i * batchsize:(i + 1) * batchsize] = find_replicant(fake_sample[i * batchsize:(i + 1) * batchsize], test)
                distance_test_TT[i * batchsize:(i + 1) * batchsize] = find_replicant_self(test[i * batchsize:(i + 1) * batchsize], test)
                distance_test_SS[i * batchsize:(i + 1) * batchsize] = find_replicant_self(fake_sample[i * batchsize:(i + 1) * batchsize], fake_sample)
                pbar.update(1)
            aa_test += (np.sum(distance_test_TS > distance_test_TT) + np.sum(distance_test_ST > distance_test_SS)) / n_test / 2
        aa_test /= n_draw

    privacy_loss = aa_test - aa_train
    return privacy_loss

def calculate_nnaa(train, test, fake, cont_cols, batchsize = 1000):
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

    result = np.abs(each_group(train.values, test.values, fake.values, n_train, n_test, batchsize))
    elapsed2 = (time.time() - start2)
    print("Risk: " + str(result) + ".")
    print("Time used: " + str(elapsed1 + elapsed2) + " seconds.")
    print("Loading time used: " + str(elapsed1) + " seconds.")
    print("Computing time used: " + str(elapsed2) + " seconds.")
    return result