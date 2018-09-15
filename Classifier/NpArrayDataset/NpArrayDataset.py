import copy
import math
import random
import time

from Classifier.AbstractDataset import AbstractDataset
import numpy as np

from Classifier.Literal import Literal
from Classifier.Rule import Rule


class NpArrayDataset(AbstractDataset):
    def __init__(self, prod=1, dataset=None, col_val_tables_pos=None, col_val_tables_neg=None, col_names=None,
                 col_unique_values=None):
        if dataset is None:
            self.prod = prod
            self.col_val_tables_pos = col_val_tables_pos
            self.col_val_tables_neg = col_val_tables_neg
            self.col_names = col_names
            self.col_unique_values = col_unique_values
        else:
            self.prod = prod
            self.col_val_tables_pos = list()
            self.col_val_tables_neg = list()
            self.col_names = dataset.columns
            self.col_names = self.col_names[:-1]
            self.col_unique_values = list()
            for i in range(0, len(self.col_names)):
                self.col_unique_values.append(dataset[self.col_names[i]].unique())
                self.col_val_tables_pos.append(list())
                self.col_val_tables_neg.append(list())
            col_val_tables_pos_tmp = list()
            col_val_tables_neg_tmp = list()
            for i in range(0, len(self.col_names)):
                col_val_tables_pos_tmp.append(list())
                col_val_tables_neg_tmp.append(list())
                for j in range(0, len(self.col_unique_values[i])):
                    col_val_tables_pos_tmp[i].append(list())
                    col_val_tables_neg_tmp[i].append(list())
            start = time.time()
            for x in dataset.iterrows():
                for i in range(0, len(self.col_names)):
                    for j in range(0, len(self.col_unique_values[i])):
                        if x[1][self.col_names[i]] == self.col_unique_values[i][j]:
                            if x[1][dataset.columns[-1]] == 1:
                                col_val_tables_pos_tmp[i][j].append(True)
                            else:
                                col_val_tables_neg_tmp[i][j].append(True)
                        else:
                            if x[1][dataset.columns[-1]] == 1:
                                col_val_tables_pos_tmp[i][j].append(False)
                            else:
                                col_val_tables_neg_tmp[i][j].append(False)
            end = time.time()
            print(end - start)
            for i in range(0, len(self.col_names)):
                for j in range(0, len(self.col_unique_values[i])):
                    self.col_val_tables_pos[i].append(np.array(col_val_tables_pos_tmp[i][j]))
                    self.col_val_tables_neg[i].append(np.array(col_val_tables_neg_tmp[i][j]))

    def delete_covered(self, rule):
        p_rule, n_rule = self.make_rules_from_iters(rule)
        idx_p = np.where(p_rule)
        idx_n = np.where(n_rule)
        for i in range(0, len(self.col_val_tables_pos)):
            for j in range(0, len(self.col_val_tables_pos[i])):
                self.col_val_tables_pos[i][j] = np.delete(self.col_val_tables_pos[i][j], idx_p)
                self.col_val_tables_neg[i][j] = np.delete(self.col_val_tables_neg[i][j], idx_n)

    def delete_not_covered(self, rule):
        p_rule, n_rule = self.make_rules_from_iters(rule)
        for i in range(0, len(self.col_val_tables_pos)):
            for j in range(0, len(self.col_val_tables_pos[i])):
                to_delete_pos = np.logical_and(self.col_val_tables_pos[i][j], p_rule)
                to_delete_neg = np.logical_and(self.col_val_tables_neg[i][j], n_rule)
                idx_p = np.where(to_delete_pos == False)
                idx_n = np.where(to_delete_neg == False)
                self.col_val_tables_pos[i][j] = np.delete(self.col_val_tables_pos[i][j], idx_p)
                self.col_val_tables_neg[i][j] = np.delete(self.col_val_tables_neg[i][j], idx_n)

    def grow_rule(self):
        best_l = None
        best_foil = -math.inf
        rule = list()
        for i in range(0, len(self.col_val_tables_neg)):
            for j in range(0, len(self.col_val_tables_neg[i])):
                p = np.count_nonzero(self.col_val_tables_pos[i][j] == True)
                n = np.count_nonzero(self.col_val_tables_neg[i][j] == True)
                tmp_foil = count_foil_grow(0, 0, p, n)
                if tmp_foil > best_foil:
                    best_l = (i, j)
                    best_foil = tmp_foil
        rule.append(best_l)
        while True:
            best_foil = -math.inf
            best_l = None
            for i in range(0, len(self.col_val_tables_neg)):
                for j in range(0, len(self.col_val_tables_neg[i])):
                    p0, n0 = self.count_p_n_rule(rule)
                    new_rule = copy.deepcopy(rule)
                    new_rule.append([i, j])
                    p, n = self.count_p_n_rule(new_rule)
                    tmp_foil = count_foil_grow(p0, n0, p, n)
                    if tmp_foil > best_foil:
                        best_foil = tmp_foil
                        best_l = (i, j)
            if best_foil > 0:
                rule.append(best_l)
            else:
                break
        return rule

    def make_rule(self, rule):
        rule = sorted(rule, key=lambda x: x[0])
        prev_i = -1
        new_rule = Rule()
        for i, j in rule:
            if prev_i != i:
                if prev_i != -1:
                    l = Literal(self.col_names[prev_i], 'in', values)
                    new_rule.add_literal(l)
                values = list()
            values.append(self.col_unique_values[i][j])
            prev_i = i
        l = Literal(self.col_names[i], 'in', values)
        new_rule.add_literal(l)
        return new_rule

    def prune_rule(self, rule):
        len_rule = len(rule) - 1
        for i in range(len_rule, -1, -1):
            pruned_rule = copy.deepcopy(rule)
            del pruned_rule[i]
            p, n = self.count_p_n_rule(rule)
            p0, n0 = self.count_p_n_rule(pruned_rule)
            if p0 != 0 and p != 0:
                if p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2)) <= 0:
                    del rule[i]
            if p == 0:
                del rule[i]
        if p == 0 or n >= p or len(rule) == 0:
            return None
        else:
            return rule

    def split_into_growset_pruneset(self):
        count_p_growset = round(len(self.col_val_tables_pos[0][0]) * 2 / 3)
        count_n_growset = round(len(self.col_val_tables_neg[0][0]) * 2 / 3)
        idx_p = random.sample(range(0, len(self.col_val_tables_pos[0][0])), count_p_growset)
        idx_n = random.sample(range(0, len(self.col_val_tables_neg[0][0])), count_n_growset)
        col_val_tables_pos_grow = list()
        col_val_tables_neg_grow = list()
        col_val_tables_pos_prune = list()
        col_val_tables_neg_prune = list()
        for i in range(0, len(self.col_names)):
            col_val_tables_pos_grow.append(list())
            col_val_tables_neg_grow.append(list())
            col_val_tables_pos_prune.append(list())
            col_val_tables_neg_prune.append(list())
            for j in range(0, len(self.col_unique_values[i])):
                col_val_tables_pos_grow[i].append(np.take(self.col_val_tables_pos[i][j], idx_p))
                col_val_tables_neg_grow[i].append(np.take(self.col_val_tables_neg[i][j], idx_n))
                col_val_tables_pos_prune[i].append(np.delete(self.col_val_tables_pos[i][j], idx_p))
                col_val_tables_neg_prune[i].append(np.delete(self.col_val_tables_neg[i][j], idx_n))
        return NpArrayDataset(prod=self.prod, col_val_tables_pos=col_val_tables_pos_grow,
                              col_val_tables_neg=col_val_tables_neg_grow,
                              col_names=self.col_names, col_unique_values=self.col_unique_values), \
               NpArrayDataset(prod=self.prod, col_val_tables_pos=col_val_tables_pos_prune,
                              col_val_tables_neg=col_val_tables_neg_prune,
                              col_names=self.col_names, col_unique_values=self.col_unique_values)

    def count_p_n_rule(self, rule):
        p_rule, n_rule = self.make_rules_from_iters(rule)
        return np.count_nonzero(p_rule == True), np.count_nonzero(n_rule == True)

    def make_rules_from_iters(self, rule):
        if rule is None:
            return None
        rule = sorted(rule, key=lambda x: x[0])
        act_rule_p = None
        act_rule_n = None
        act_rule_n_tmp = None
        act_rule_p_tmp = None
        prev_i = -1
        for i, j in rule:
            if prev_i != i:
                if act_rule_p is None:
                    act_rule_p = act_rule_p_tmp
                    act_rule_n = act_rule_n_tmp
                else:
                    act_rule_p = np.logical_and(act_rule_p, act_rule_p_tmp)
                    act_rule_n = np.logical_and(act_rule_n, act_rule_n_tmp)
                act_rule_p_tmp = self.col_val_tables_pos[i][j]
                act_rule_n_tmp = self.col_val_tables_neg[i][j]
            else:
                act_rule_p_tmp = np.logical_or(act_rule_p_tmp, self.col_val_tables_pos[i][j])
                act_rule_n_tmp = np.logical_or(act_rule_n_tmp, self.col_val_tables_neg[i][j])
            prev_i = i
        if act_rule_p is None:
            act_rule_p = act_rule_p_tmp
            act_rule_n = act_rule_n_tmp
        else:
            act_rule_p = np.logical_and(act_rule_p, act_rule_p_tmp)
            act_rule_n = np.logical_and(act_rule_n, act_rule_n_tmp)
        return act_rule_p, act_rule_n

    def is_any_pos_example(self):
        return len(self.col_val_tables_pos[0][0]) > 0

    def length(self):
        return len(self.col_val_tables_pos[0][0]) + len(self.col_val_tables_neg[0][0])


def count_foil_grow(p0, n0, p, n):
    if p0 == 0 and n0 == 0:
        if p == 0:
            return -math.inf
        try:
            return p * (p / (p + n))
        except (ZeroDivisionError, ValueError):
            return -math.inf
    else:
        if p == 0:
            return -math.inf
        if n == 0 and n0 == 0:
            return p - p0
        try:
            return p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2))
        except (ZeroDivisionError, ValueError):
            return -math.inf
