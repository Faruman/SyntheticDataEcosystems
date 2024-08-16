import numpy as np
import time
from scipy import stats
import os.path
import sys
from tqdm import trange

def get_entropy(column):
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk = hist1 / len(column)
    entropy = -np.sum(pk * np.log(pk))
    return entropy

def find_neighbour(r, r_, data, data_, k, cont_sense_attr):
    # k: k nearest neighbours
    diff_array = np.abs(data - r)
    diff_array_max = np.amax(diff_array, axis=0)
    diff_array_max2 = np.maximum(diff_array_max, 1)
    diff_array_rate = diff_array/diff_array_max2
    diff = np.sum(diff_array_rate, axis=1)
    thresh = np.sort(diff)[k-1]
    idxs = np.arange(len(data))[diff <= thresh]  # not exactly k neighbours?
    predict = stats.mode(data_[idxs])[0][0]

    if len(cont_sense_attr) > 0:
        cont_sense_attr_mask = np.ones_like(r_, bool)
        cont_sense_attr_mask[cont_sense_attr] = False
        bin_r_ = r_[cont_sense_attr_mask]
        bin_predict = predict[cont_sense_attr_mask]
        cont_r_ = r_[cont_sense_attr]
        cont_predict = predict[cont_sense_attr]
        bin_n = len(bin_r_)  # number of binary attributes
        true_pos = ((bin_predict + bin_r_) == 2)
        false_pos = np.array([(bin_r_[i] == 0) and (bin_predict[i] == 1) for i in range(bin_n)])
        false_neg = np.array([(bin_r_[i] == 1) and (bin_predict[i] == 0) for i in range(bin_n)])
        correct_cont_predict = np.logical_and(cont_predict <= cont_r_ * 1.1, cont_predict >= cont_r_ * 0.9)
    else:
        bin_n = len(r_)  # number of binary attributes
        true_pos = ((predict + r_) == 2)
        false_pos = np.array([(r_[i] == 0) and (predict[i] == 1) for i in range(bin_n)])
        false_neg = np.array([(r_[i] == 1) and (predict[i] == 0) for i in range(bin_n)])
        correct_cont_predict = 0
    return true_pos, false_pos, false_neg, correct_cont_predict

class Model(object):
    def __init__(self, fake, n, k, sensitive_cols_num, attr_cols_num, cont_cols_num):
        self.fake = fake
        self.n = n  # number of attributes used by the attacker
        self.k = k  # k nearest neighbours
        self.true_pos = []
        self.false_pos = []
        self.false_neg = []
        self.attr_idx = attr_cols_num  # selected attributes' indexes
        self.attr_idx_ = sensitive_cols_num # unselected attributes' indexes
        self.data = self.fake[:, self.attr_idx]
        self.data_ = self.fake[:, self.attr_idx_]
        self.cont_cols_num = cont_cols_num
        if len(cont_cols_num) > 0:
            self.correct = []
            self.cont_sense_attr = np.searchsorted(sensitive_cols_num,np.array(list(set(cont_cols_num).intersection(set(sensitive_cols_num)))))

    def single_r(self, R):
        r = R[self.attr_idx]  # tested record's selected attributes
        r_ = R[self.attr_idx_]  # tested record's unselected attributes
        if len(self.cont_cols_num ) > 0:
            true_pos, false_pos, false_neg, correct = find_neighbour(r, r_, self.data, self.data_, self.k, self.cont_sense_attr)
            self.correct.append(correct)
        else:
            true_pos, false_pos, false_neg, _ = find_neighbour(r, r_, self.data, self.data_, self.k, 0)
        self.true_pos.append(true_pos)
        self.false_pos.append(false_pos)
        self.false_neg.append(false_neg)

def cal_score(n, k, real, fake, sensitive_cols_num, attr_cols_num, cont_cols_num):
    # 2^n: the number of attributes used by the attacker
    # 10^k: the number of neighbours

    model = Model(fake, 2 ** n, 10 ** k, sensitive_cols_num, attr_cols_num, cont_cols_num)
    n_rows = np.shape(real)[0]
    for i in trange(n_rows, desc= "Calculating Score"):
        record = real[i, :]
        model.single_r(record)

    # binary part
    tp_array = np.stack(model.true_pos, axis=0)  # array of true positives
    fp_array = np.stack(model.false_pos, axis=0)  # array of false positives
    fn_array = np.stack(model.false_neg, axis=0)  # array of false negatives
    tpc = np.sum(tp_array, axis=0)  # vector of true positive count
    fpc = np.sum(fp_array, axis=0)  # vector of false positive count
    fnc = np.sum(fn_array, axis=0)  # vector of false negative count
    f1 = np.nan_to_num(tpc / (tpc + 0.5 * (fpc + fnc)))

    # continuous part
    if len(cont_cols_num) > 0:
        correct_array = np.stack(model.correct, axis=0)  # array of correctness
        accuracy = np.mean(correct_array, axis=0)

    # compute weights
    entropy = []
    real_ = real[:, model.attr_idx_]
    n_attr_ = np.shape(real_)[1]  # number of predicted attributes
    for j in range(n_attr_):
        entropy.append(get_entropy(real_[:, j]))
    weight = np.asarray(entropy) / sum(entropy)
    if len(cont_cols_num) > 0:
        bin_weight = weight[np.logical_not(model.cont_sense_attr)]
        cont_weight = weight[model.cont_sense_attr]
        score = np.sum(np.concatenate([f1, accuracy]) * np.concatenate([bin_weight, cont_weight]))
    else:
        score = np.sum(f1 * weight)
    return score

def calculate_air(train, test, fake, cont_cols, sensitive_cols, x= 0, y= 8, batchsize=1000):
    start1 = time.time()

    if len(cont_cols) > 0:
        cont_cols_num = [i for i in range(len(fake.columns)) if fake.columns[i] in cont_cols]
    else:
        cont_cols_num = []

    sensitive_cols_num = [i for i in range(len(fake.columns)) if fake.columns[i] in sensitive_cols]
    attr_cols_num = [i for i in range(len(fake.columns)) if fake.columns[i] not in sensitive_cols]

    result = cal_score(y, x, train.values, fake.values, sensitive_cols_num, attr_cols_num, cont_cols_num)
    elapsed1 = (time.time() - start1)
    print("Risk: " + str(result) + ".")
    print("Time used: " + str(elapsed1) + " seconds.")
    return result