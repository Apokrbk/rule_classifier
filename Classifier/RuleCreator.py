from Classifier.DictDataset import DictDataset
import pandas as pd
import random

import copy
import time

def create_rules(trainset_in):
    rules = list()
    max_iter = 0
    trainset = DictDataset(trainset_in)
    while True:
        growset, pruneset = trainset.split_into_growset_pruneset()
        start = time.time()
        new_rule = growset.grow_rule()
        new_rule = pruneset.prune_rule(new_rule)
        end = time.time()
        p, n = pruneset.count_p_n_rule(new_rule)
        if len(new_rule.literals) == 0 or n >= p or p == 0:
            print("BAD RULE " + "Time: " + str(end - start) + "s")
            max_iter += 1
        else:
            trainset.delete_covered(new_rule)
            rule_to_add = copy.deepcopy(new_rule)
            rules.append(copy.deepcopy(rule_to_add))
            print("Rule: " + new_rule.to_string() + "Time: " + str(end - start) + "s")
        if max_iter >= 3 or trainset.length() < 60 or not trainset.is_any_pos_example():
            break
    return rules


def test_all(df):
    last_col_name = df.columns[len(df.columns) - 1]
    all_ex = len(df)
    p_ex = df[df.columns[len(df.columns) - 1]].sum()
    n_ex = all_ex - p_ex
    print(all_ex, p_ex, n_ex)
    start = time.time()
    rules = create_rules(df)
    end = time.time()
    p_all = 0
    n_all = 0
    for i in range(0, len(rules)):
        print(rules[i].to_string())
        print(rules[i].count_p_n(df, last_col_name))
        p, n = rules[i].count_p_n(df, last_col_name)
        p_all += p
        n_all += n
        df = delete_covered(df, rules[i])
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
    print("Full time: " + str(end - start))


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


df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/titanic3.csv', encoding='utf-8', delimiter=',')
test_all(df)
