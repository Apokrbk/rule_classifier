import time

from Classifier.literal import Literal
from Classifier.rule import Rule
import copy
import math
import random
from Classifier.abstract_datasets.abstract_dataset import AbstractDataset, count_foil_grow
import numpy as np
import pandas as pd



class NpArrayDataset(AbstractDataset):
    def __init__(self, prod=1, dataset=None, col_val_tables=None, col_names=None,
                 col_unique_values=None, pos_map=None, neg_map=None):
        super().__init__(prod, dataset)
        if dataset is None:
            self.prod = prod
            self.col_val_tables = col_val_tables
            self.col_names = col_names
            self.col_unique_values = col_unique_values
            self.pos_map = pos_map
            self.neg_map = neg_map
        else:
            self.prod = prod
            self.col_val_tables = list()
            self.col_names = dataset.columns
            self.col_names = self.col_names[:-1]
            self.col_unique_values = list()
            self.create_tables_for_every_value(dataset)
            self.pos_map = np.array(dataset[dataset.columns[-1]].map(lambda x: x == 1))
            self.neg_map = np.logical_not(self.pos_map)
            if len(dataset) != 0:
                for i in range(0, len(self.col_names)):
                    for j in range(0, len(self.col_unique_values[i])):
                        act_col = self.col_names[i]
                        act_value = self.col_unique_values[i][j]
                        self.col_val_tables[i].append(NpArrayDataset.create_bitmap_for_value(act_col, act_value, dataset))

    def create_tables_for_every_value(self, dataset):
        for i in range(0, len(self.col_names)):
            self.col_unique_values.append(dataset[self.col_names[i]].unique())
            self.col_val_tables.append(list())

    @staticmethod
    def create_bitmap_for_value(act_col, act_value, dataset):
        return np.array(dataset[act_col].map(lambda x: x == act_value))

    def delete_covered(self, rule):
        new_rule = self.make_rules_from_iters(rule)
        idx = np.where(new_rule)
        for i in range(0, len(self.col_val_tables)):
            for j in range(0, len(self.col_val_tables[i])):
                self.col_val_tables[i][j] = np.delete(self.col_val_tables[i][j], idx)
        self.pos_map = np.delete(self.pos_map, idx)
        self.neg_map = np.delete(self.neg_map, idx)

    def delete_not_covered(self, rule):
        new_rule = self.make_rules_from_iters(rule)
        idx = np.where(new_rule)
        for i in range(0, len(self.col_val_tables)):
            for j in range(0, len(self.col_val_tables[i])):
                self.col_val_tables[i][j] = np.take(self.col_val_tables[i][j], idx)[0]
        self.pos_map = np.take(self.pos_map, idx)[0]
        self.neg_map = np.take(self.neg_map, idx)[0]

    def grow_rule(self):
        return self.grow_rule_sorted_p_n()

        # return self.grow_rule_inductive()

    def grow_rule_inductive(self):
        best_rule = list()
        while True:
            best_foil = -math.inf
            best_l = None
            for i in range(0, len(self.col_val_tables)):
                for j in range(0, len(self.col_val_tables[i])):
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

    def grow_rule_sorted_p_n(self):
        best_rule = list()
        while True:
            best_l = None
            best_foil = -math.inf
            p0, n0 = self.count_p_n_rule(best_rule)
            for i in range(0, len(self.col_names)):
                tmp_l, tmp_foil = self.find_best_literal_from_variable(i, p0, n0, best_rule)
                if tmp_foil > best_foil:
                    best_l = copy.deepcopy(tmp_l)
                    best_foil = tmp_foil
            if best_foil <= 0:
                break
            else:
                best_rule = best_rule + best_l
        return best_rule

    def find_best_literal_from_variable(self, var, p0, n0, old_rule):
        best_foil = -math.inf
        best_l = None
        p_to_n = self.count_p_n_for_every_value(old_rule, var)
        new_literal = list()
        best_foil, best_l = self.choose_best_literal(best_foil, best_l, n0, new_literal, old_rule, p0, p_to_n, var)
        if best_foil <= 0:
            return list(), -math.inf
        else:
            return best_l, best_foil

    def choose_best_literal(self, best_foil, best_l, n0, new_literal, old_rule, p0, p_to_n, var):
        for i in range(0, len(p_to_n)):
            new_literal.append([var, p_to_n[i][0]])
            new_rule = old_rule + new_literal
            p, n = self.count_p_n_rule(new_rule)
            foil = count_foil_grow(p0, n0, p, n)
            if foil > best_foil:
                best_foil = foil
                best_l = copy.deepcopy(new_literal)
            else:
                break
        return best_foil, best_l

    def count_p_n_for_every_value(self, old_rule, var):
        p_to_n = list()
        for i in range(0, len(self.col_val_tables[var])):
            new_literal = list()
            new_literal.append([var, i])
            new_rule = old_rule + new_literal
            p, n = self.count_p_n_rule(new_rule)
            if n == 0:
                p_to_n.append([i, math.inf])
            else:
                p_to_n.append([i, p / n])
        p_to_n = sorted(p_to_n, key=lambda x: x[1], reverse=True)
        return p_to_n

    def make_rule(self, rule):
        if len(rule) == 0:
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
        if len(rule.literals) == 0:
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
        if len(rule) == 0:
            return None
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
        count_growset = round(len(self.col_val_tables[0][0]) * 2 / 3)
        idx = self.choose_idx_for_split(count_growset)
        col_val_tables_grow, col_val_tables_prune = self.split_by_idx(idx)
        pos_map_grow = np.take(self.pos_map, idx)
        neg_map_grow = np.take(self.neg_map, idx)
        pos_map_prune = np.delete(self.pos_map, idx)
        neg_map_prune = np.delete(self.neg_map, idx)
        return NpArrayDataset(prod=self.prod, col_val_tables=col_val_tables_grow,
                              col_names=self.col_names, col_unique_values=self.col_unique_values, pos_map=pos_map_grow,
                              neg_map=neg_map_grow), \
               NpArrayDataset(prod=self.prod, col_val_tables=col_val_tables_prune,
                              col_names=self.col_names, col_unique_values=self.col_unique_values, pos_map=pos_map_prune,
                              neg_map=neg_map_prune)

    def choose_idx_for_split(self, count_growset):
        if self.prod == 1:
            idx = random.sample(range(0, len(self.col_val_tables[0][0])), count_growset)
        else:
            idx = range(0, count_growset)
        return idx

    def split_by_idx(self, idx):
        col_val_tables_grow = list()
        col_val_tables_prune = list()
        for i in range(0, len(self.col_names)):
            col_val_tables_grow.append(list())
            col_val_tables_prune.append(list())
            for j in range(0, len(self.col_unique_values[i])):
                col_val_tables_grow[i].append(np.take(self.col_val_tables[i][j], idx))
                col_val_tables_prune[i].append(np.delete(self.col_val_tables[i][j], idx))
        return col_val_tables_grow, col_val_tables_prune

    def count_p_n_rule(self, rule):
        if len(rule) == 0:
            return np.count_nonzero(self.pos_map == True), np.count_nonzero(self.neg_map == True)
        else:
            new_rule = self.make_rules_from_iters(rule)
            p_rule = np.logical_and(new_rule, self.pos_map)
            n_rule = np.logical_and(new_rule, self.neg_map)
            return np.count_nonzero(p_rule == True), np.count_nonzero(n_rule == True)

    def make_rules_from_iters(self, rule):
        rule = sorted(rule, key=lambda x: x[0])
        act_rule = None
        act_rule_tmp = None
        prev_i = -1
        for i, j in rule:
            if prev_i != i:
                if act_rule is None:
                    act_rule = act_rule_tmp
                else:
                    act_rule = np.logical_and(act_rule, act_rule_tmp)
                act_rule_tmp = self.col_val_tables[i][j]
            else:
                act_rule_tmp = np.logical_or(act_rule_tmp, self.col_val_tables[i][j])
            prev_i = i
        if act_rule is None:
            act_rule = act_rule_tmp
        else:
            act_rule = np.logical_and(act_rule, act_rule_tmp)
        return act_rule

    def is_any_pos_example(self):
        return np.count_nonzero(self.pos_map == True) > 0

    def length(self):
        return len(self.pos_map)


