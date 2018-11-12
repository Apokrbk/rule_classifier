import copy
import math
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import time

from Classifier.abstract_datasets.bitmap_dataset.bitmap_dataset import BitmapDataset
from Classifier.abstract_datasets.dict_dataset.dict_dataset import DictDataset
from Classifier.abstract_datasets.nparray_dataset.nparray_dataset import NpArrayDataset


class RuleCreator:

    def __init__(self, dataset_type, df, prod):
        self.prod = prod
        self.rules = self.train(dataset_type(prod, df))

    def train(self, trainset):
        rules = list()
        max_iter = 0
        while True:
            growset, pruneset = trainset.split_into_growset_pruneset()
            start = time.time()
            new_rule = growset.grow_rule()
            new_rule = pruneset.prune_rule(new_rule)
            end = time.time()
            if new_rule is None:
                # print("BAD RULE " + "Time: " + str(end - start) + "s")
                max_iter += 1
            else:
                trainset.delete_covered(new_rule)
                new_rule = trainset.make_rule(new_rule)
                rules.append(new_rule)
                # print("Rule: " + new_rule.to_string() + "Time: " + str(end - start) + "s")
            if max_iter >= 5 or trainset.length() < 60 or not trainset.is_any_pos_example():
                break
        return rules

    def predict(self, dataset):
        predictions = list()
        for index, row in dataset.iterrows():
            if self.row_covered(row):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def row_covered(self, row):
        if len(self.rules) == 0:
            return True
        for i in range(0, len(self.rules)):
            if self.rules[i].row_covered(row):
                return True
        return False

    def get_rules(self):
        return self.rules

    def get_number_of_rules(self):
        return len(self.rules)

    def print_rules(self):
        for i in range(0, len(self.rules)):
            print(self.rules[i].to_string())


def split_into_trainset_testset(df, ratio):
    df = df.sample(frac=1)
    df.index = range(len(df))
    div_idx = math.floor(len(df) * ratio)
    trainset = df[0:div_idx]
    trainset.index = range(len(trainset))
    testset = df[div_idx:]
    testset.index = range(len(testset))
    return trainset, testset


def cubes_for_numeric_data(df, num_of_intervals):
    numeric_cols = df._get_numeric_data().columns
    numeric_cols = numeric_cols[:-1]
    for i in range(0, len(numeric_cols)):
        df[numeric_cols[i]] = pd.qcut(df[numeric_cols[i]], num_of_intervals, duplicates='drop').astype(str)
    return df


def remove_empty_values(df):
    numeric_cols = df._get_numeric_data().columns
    char_cols = df.select_dtypes('object').columns
    for i in range(0, len(numeric_cols)):
        avg = df[numeric_cols[i]].mean()
        df[numeric_cols[i]] = df[numeric_cols[i]].fillna(value=avg)
    for i in range(0, len(char_cols)):
        df[char_cols[i]] = df[char_cols[i]].fillna(value='null')
    return df


def count_tp_fp_tn_fn(classes, predictions):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if len(classes) != len(predictions):
        print("Vector with classes and vector with predictions should have same lenght")
        return
    for i in range(0, len(classes)):
        if classes[i] == predictions[i]:
            if classes[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if classes[i] == 1:
                fn += 1
            else:
                fp += 1
    return tp, fp, tn, fn


def test_all(df_all, prod, iters, dataset_type, filename):
    df_train, df_test = split_into_trainset_testset(df_all, 0.8)
    all_ex = len(df_test)
    p_ex = df_test[df_test.columns[-1]].sum()
    n_ex = all_ex - p_ex
    tp = list()
    fp = list()
    tn = list()
    fn = list()
    times = list()
    number_of_rules = list()
    errors = list()
    p = list()
    n = list()
    all = list()
    features = list()
    for j in range(0, iters):
        p.append(p_ex)
        n.append(n_ex)
        all.append(all_ex)
        start = time.time()
        rule_creator = RuleCreator(dataset_type, df_train, prod)
        end = time.time()
        times.append(end - start)
        number_of_rules.append(rule_creator.get_number_of_rules())
        predictions = rule_creator.predict(df_test)
        tp_tmp, fp_tmp, tn_tmp, fn_tmp = count_tp_fp_tn_fn(df_test[df_test.columns[-1]].tolist(), predictions)
        tp.append(tp_tmp)
        fp.append(fp_tmp)
        tn.append(tn_tmp)
        fn.append(fn_tmp)
        errors.append(fp_tmp + fn_tmp)
        features.append(len(df_train.columns) - 1)
        print(j)
    results = pd.DataFrame(
        {'All examples': all,
         'Positive examples': p,
         'Negative examples': n,
         'TP': tp,
         'FP': fp,
         'TN': tn,
         'FN': fn,
         'Errors (FP + FN)': errors,
         'Time in seconds': times,
         'Number of rules': number_of_rules,
         'Number of features': features
         })
    results.to_csv(filename, sep=';', encoding='utf-8')
    return results


# print("MUSHROOM")
# df = pd.read_csv('data_files/mushroom.csv',
#                  encoding='utf-8', delimiter=';')
# rule_creator = RuleCreator(NpArrayDataset, df, 1)
# predictions = rule_creator.predict(df)
# tp, fp, tn, fn = count_tp_fp_tn_fn(df[df.columns[-1]].tolist(), predictions)
# print(tp,fp,tn,fn)

#
print("MUSHROOM")
# df = pd.read_csv('data_files/hypothyroid.csv',
#                  encoding='utf-8', delimiter=';')
df = pd.read_csv('data_files/hypothyroid_mult1000.csv',
                 encoding='utf-8', delimiter=';')
df = cubes_for_numeric_data(df, 10)
# df = remove_empty_values(df)
test_all(df, 1, 1, BitmapDataset, 'results_files/hypothyroid_mult1000_bitmap.csv')
#
# print("MUSHROOM")
# df = pd.read_csv('data_files/bank-full.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df, 10)
# test_all(df, 1, 1, NpArrayDataset)
# print("HYPOTHYROID")
# df = pd.read_csv('data_files/hypothyroid.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,10)
# test_all(df, 1, 10, NpArrayDataset)
#
# print("HYPOTHYROID")
# df = pd.read_csv('data_files/hypothyroid.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,30)
# test_all(df, 1, 10, NpArrayDataset)
#
# print("HYPOTHYROID")
# df = pd.read_csv('data_files/hypothyroid.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,10)
# test_all(df, 1, 10, NpArrayDataset)

# print("HYPOTHYROID")
# df = pd.read_csv('data_files/hypothyroid.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,10)
# test_all(df, 1, 10, BitmapDataset)
#
# print("HYPOTHYROID")
# df = pd.read_csv('data_files/hypothyroid.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,50)
# test_all(df, 1, 10, NpArrayDataset)

# print("PHONEME")
# df = pd.read_csv('data_files/phoneme.csv',
#                  encoding='utf-8', delimiter=';')
# # df = cubes_for_numeric_data(df,10)
# test_all(df, 1, 3, DictDataset)
#
# print("PHONEME")
# df = pd.read_csv('data_files/phoneme.csv',
#                  encoding='utf-8', delimiter=';')
# test_all(df, 1, 10, DictDataset, 'results_files/phoneme_dictdataset.csv')

# print("PHONEME")
# df = pd.read_csv('data_files/phoneme.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,30)
# test_all(df, 1, 1, BitmapDataset)
#
# print("PHONEME")
# df = pd.read_csv('data_files/phoneme.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,50)
# test_all(df, 1, 10, NpArrayDataset)
