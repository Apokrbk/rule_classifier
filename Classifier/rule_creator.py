import copy
import math
import numpy as np
import pandas as pd

import time

from Classifier.abstract_datasets.bitmap_dataset.bitmap_dataset import BitmapDataset
from Classifier.abstract_datasets.dict_dataset.dict_dataset import DictDataset
from Classifier.abstract_datasets.nparray_dataset.nparray_dataset import NpArrayDataset


def create_rules(trainset):
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
        if max_iter >= 3 or trainset.length() < 60 or not trainset.is_any_pos_example():
            break
    return rules


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
        np_array = np.array(df[numeric_cols[i]])
        interval_p = 100 / num_of_intervals
        df.insert(0, numeric_cols[i] + "__", None)
        for j in range(0, num_of_intervals):
            if j == num_of_intervals - 1:
                perc = np.percentile(np_array, j * interval_p)
                df.loc[(df[numeric_cols[i]] >= perc), numeric_cols[i] + "__"] = str("P" + str(j))
            elif j == 0:
                perc = np.percentile(np_array, interval_p)
                df.loc[(df[numeric_cols[i]] < perc), numeric_cols[i] + "__"] = str("P" + str(j))
            else:
                perc_from = np.percentile(np_array, j * interval_p)
                perc_to = np.percentile(np_array, (j + 1) * interval_p)
                df.loc[
                    (df[numeric_cols[i]] >= perc_from) & (df[numeric_cols[i]] < perc_to), numeric_cols[i] + "__"] = str(
                    "P" + str(j))
        df = df.drop(numeric_cols[i], axis=1)
        df = df.rename(index=str, columns={numeric_cols[i] + "__": numeric_cols[i]})
    return df


def test_all(df_all, prod, iters, dataset_type):
    df_train, df = split_into_trainset_testset(df_all, 0.8)
    all_ex = len(df)
    p_ex = df[df.columns[-1]].sum()
    n_ex = all_ex - p_ex
    p_ex *= iters
    n_ex *= iters
    all_ex *= iters
    print(all_ex, p_ex, n_ex)
    p_all = 0
    n_all = 0
    time_all = 0
    rules_all = 0
    for j in range(0, iters):
        c_df = copy.deepcopy(df)
        start = time.time()
        dataset = dataset_type(prod, df_train)
        rules = create_rules(dataset)
        end = time.time()
        for i in range(0, len(rules)):
            # print(rules[i].to_string())
            # print(rules[i].count_p_n(df, last_col_name))
            p, n = rules[i].count_p_n(c_df)
            p_all += p
            n_all += n
            c_df = delete_covered(c_df, rules[i])
        time_all += (end - start)
        rules_all += len(rules)
    tp = p_all
    fp = n_all
    tn = n_ex - n_all
    fn = p_ex - p_all
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    print("TP: " + str(tp) + " FP: " + str(fp))
    print("TN: " + str(tn) + " FN: " + str(fn))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))
    print("Average time: " + str(time_all / iters))
    print("Average number of rules: " + str(rules_all / iters))
    print("Average errors: " + str((fp + fn) / iters))


def delete_covered(growset, rule):
    growset_dict = growset.to_dict()
    idx = set()
    for i in range(0, len(growset)):
        to_delete = True
        for j in range(0, len(rule.literals)):
            if not rule.literals[j].value_covered_by_literal(growset_dict[rule.literals[j].var_name][i]):
                to_delete = False
                break
        if to_delete:
            idx.add(i)
    indexes_to_keep = set(range(growset.shape[0])) - idx
    growset = growset.take(list(indexes_to_keep))
    growset.index = range(len(growset))
    return growset





print("MUSHROOM")
df = pd.read_csv('data_files/mushroom.csv',
                 encoding='utf-8', delimiter=';')
test_all(df, 1, 1, DictDataset)

# print("HYPOTHYROID")
# df = pd.read_csv('data_files/hypothyroid.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,10)
# test_all(df, 1, 10)

# print("PHONEME")
# df = pd.read_csv('data_files/phoneme.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,10)
# test_all(df, 1, 10)
