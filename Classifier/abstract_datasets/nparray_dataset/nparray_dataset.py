import time

from Classifier.literal import Literal
from Classifier.rule import Rule
import copy
import math
import random
from Classifier.abstract_datasets.abstract_dataset import AbstractDataset
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


class NpArrayDataset(AbstractDataset):
    def __init__(self, prod=1, dataset=None, col_val_tables_pos=None, col_val_tables_neg=None, col_names=None,
                 col_unique_values=None):
        super().__init__(prod, dataset)
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
            pos = dataset.loc[dataset[dataset.columns[-1]] == 1]
            neg = dataset.loc[dataset[dataset.columns[-1]] == 0]
            for i in range(0, len(self.col_names)):
                for j in range(0, len(self.col_unique_values[i])):
                    if len(pos) != 0:
                        pos.loc[(pos[self.col_names[i]] == self.col_unique_values[i][j]), "__temp__"] = True
                        pos.loc[(pos[self.col_names[i]] != self.col_unique_values[i][j]), "__temp__"] = False
                    else:
                        pos["__temp__"] = 1
                    if len(neg) != 0:
                        neg.loc[(neg[self.col_names[i]] == self.col_unique_values[i][j]), "__temp__"] = True
                        neg.loc[(neg[self.col_names[i]] != self.col_unique_values[i][j]), "__temp__"] = False
                    else:
                        neg["__temp__"] = 1
                    self.col_val_tables_pos[i].append(np.array(pos['__temp__'].values))
                    self.col_val_tables_neg[i].append(np.array(neg['__temp__'].values))

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
        idx_p = np.where(p_rule)
        idx_n = np.where(n_rule)
        for i in range(0, len(self.col_val_tables_pos)):
            for j in range(0, len(self.col_val_tables_pos[i])):
                self.col_val_tables_pos[i][j] = np.take(self.col_val_tables_pos[i][j], idx_p)[0]
                self.col_val_tables_neg[i][j] = np.take(self.col_val_tables_neg[i][j], idx_n)[0]

    def grow_rule(self):
        # best_rule = list()
        # best_foil = -math.inf
        # for i in range(0, len(self.col_names)):
        #     tmp_l, tmp_foil = self.find_best_literal_from_variable(i, 0, 0, best_rule)
        #     if tmp_foil > best_foil:
        #         best_rule = copy.deepcopy(tmp_l)
        #         best_foil = tmp_foil
        # if best_foil == -math.inf:
        #     return best_rule
        # while True:
        #     best_l = None
        #     best_foil = -math.inf
        #     p0,n0 = self.count_p_n_rule(best_rule)
        #     for i in range(0, len(self.col_names)):
        #         tmp_l, tmp_foil = self.find_best_literal_from_variable(i, p0,n0, best_rule)
        #         if tmp_foil>best_foil:
        #             best_l = copy.deepcopy(tmp_l)
        #             best_foil = tmp_foil
        #     if best_foil <= 0 or best_foil == -math.inf:
        #         break
        #     else:
        #         best_rule = best_rule + best_l
        #
        # return best_rule

        # best_rule=list()
        # best_l = None
        # best_foil = -math.inf
        # p0,n0 = self.count_p_n_rule(best_rule)
        # for i in range(0, len(self.col_val_tables_neg)):
        #     for j in range(0, len(self.col_val_tables_neg[i])):
        #         p = np.count_nonzero(self.col_val_tables_pos[i][j] == True)
        #         n = np.count_nonzero(self.col_val_tables_neg[i][j] == True)
        #         tmp_foil = count_foil_grow(p0, n0, p, n)
        #         if tmp_foil > best_foil:
        #             best_l = (i, j)
        #             best_foil = tmp_foil
        # if best_l is None or best_foil <= 0:
        #     return best_rule
        # best_rule.append(best_l)
        best_rule = list()
        while True:
            best_foil = -math.inf
            best_l = None
            for i in range(0, len(self.col_val_tables_neg)):
                for j in range(0, len(self.col_val_tables_neg[i])):
                    p0, n0 = self.count_p_n_rule(best_rule)
                    new_rule = copy.deepcopy(best_rule)
                    new_rule.append([i, j])
                    p, n = self.count_p_n_rule(new_rule)
                    tmp_foil = count_foil_grow(p0, n0, p, n)
                    if tmp_foil > best_foil:
                        best_foil = tmp_foil
                        best_l = (i, j)
            if best_foil > 0:
                best_rule.append(best_l)
            else:
                break
        return best_rule

    def find_best_literal_from_variable(self, var, p0, n0, old_rule):
        best_foil = -math.inf
        p_to_n = list()
        best_l = None
        for i in range(0, len(self.col_val_tables_pos[var])):
            new_literal = list()
            new_literal.append([var, i])
            new_rule = old_rule + new_literal
            p, n = self.count_p_n_rule(new_rule)
            if n == 0:
                p_to_n.append([i, math.inf])
            else:
                p_to_n.append([i, p / n])
        p_to_n = sorted(p_to_n, key=lambda x: x[1], reverse=True)
        new_literal = list()
        for i in range(0, len(p_to_n)):
            new_literal.append([var, p_to_n[i][0]])
            new_rule = old_rule + new_literal
            p, n = self.count_p_n_rule(new_rule)
            foil = count_foil_grow(p0, n0, p, n)
            if foil > best_foil:
                best_foil = foil
                best_l = copy.deepcopy(new_literal)
        if best_foil == -math.inf:
            return None, best_foil
        else:
            return best_l, best_foil

    def make_rule(self, rule):
        if len(rule)==0:
            return Rule()
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
        l = Literal(self.col_names[i], 'in', sorted(values))
        new_rule.add_literal(l)
        return new_rule

    def unmake_rule(self, rule):
        if len(rule.literals)==0:
            return list()
        new_rule = list()
        for i in range(0, len(rule.literals)):
            for j in range(0, len(self.col_names)):
                if rule.literals[i].var_name == self.col_names[j]:
                    col = j
                    break
            for j in range(0, len(self.col_unique_values[col])):
                if rule.literals[i].value_covered_by_literal(self.col_unique_values[col][j]):
                    new_rule.append([col, j])
        return new_rule

    def prune_rule(self, rule):
        len_rule = len(rule) - 1
        for i in range(len_rule, -1, -1):
            pruned_rule = copy.deepcopy(rule)
            del pruned_rule[i]
            p, n = self.count_p_n_rule(rule)
            p0, n0 = self.count_p_n_rule(pruned_rule)
            if p0 != 0 and p != 0:
                if n0 == 0 and n == 0 and p > p0:
                    pass
                elif p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2)) <= 0:
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
        if self.prod == 1:
            idx_p = random.sample(range(0, len(self.col_val_tables_pos[0][0])), count_p_growset)
            idx_n = random.sample(range(0, len(self.col_val_tables_neg[0][0])), count_n_growset)
        else:
            idx_p = range(0, count_p_growset)
            idx_n = range(0, count_n_growset)
        col_val_tables_neg_grow, col_val_tables_neg_prune, col_val_tables_pos_grow, col_val_tables_pos_prune = self.split_by_idx(
            idx_n, idx_p)
        return NpArrayDataset(prod=self.prod, col_val_tables_pos=col_val_tables_pos_grow,
                              col_val_tables_neg=col_val_tables_neg_grow,
                              col_names=self.col_names, col_unique_values=self.col_unique_values), \
               NpArrayDataset(prod=self.prod, col_val_tables_pos=col_val_tables_pos_prune,
                              col_val_tables_neg=col_val_tables_neg_prune,
                              col_names=self.col_names, col_unique_values=self.col_unique_values)

    def split_by_idx(self, idx_n, idx_p):
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
        return col_val_tables_neg_grow, col_val_tables_neg_prune, col_val_tables_pos_grow, col_val_tables_pos_prune

    def count_p_n_rule(self, rule):
        if len(rule) == 0:
            return len(self.col_val_tables_pos[0][0]), len(self.col_val_tables_neg[0][0])
        else:
            p_rule, n_rule = self.make_rules_from_iters(rule)
            return np.count_nonzero(p_rule == True), np.count_nonzero(n_rule == True)

    def make_rules_from_iters(self, rule):
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
        if len(self.col_val_tables_pos[0]) == 0:
            return False
        return len(self.col_val_tables_pos[0][0]) > 0

    def length(self):
        if len(self.col_val_tables_pos[0]) == 0 and len(self.col_val_tables_neg[0]) == 0:
            return 0
        elif len(self.col_val_tables_pos[0]) == 0:
            return len(self.col_val_tables_neg[0][0])
        elif len(self.col_val_tables_neg[0]) == 0:
            return len(self.col_val_tables_pos[0][0])
        else:
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
