import time

from Classifier.literal import Literal
from Classifier.rule import Rule
import copy
import math
import random
from Classifier.abstract_datasets.abstract_dataset import AbstractDataset, count_foil_grow
import numpy as np
from pyroaring import BitMap
import pandas as pd


class BitmapDataset(AbstractDataset):
    def __init__(self, prod=1, dataset=None, col_val_tables=None, col_names=None,
                 col_unique_values=None, pos_map=None, neg_map=None, grow_param_raw=0, prune_param_raw=0):
        super().__init__(prod, dataset)
        if dataset is None:
            self.prod = prod
            self.col_val_tables = col_val_tables
            self.col_names = col_names
            self.col_unique_values = col_unique_values
            self.pos_map = pos_map
            self.neg_map = neg_map
            self.all_id = self.pos_map | self.neg_map
            self.grow_param_raw = grow_param_raw
            self.prune_param_raw = prune_param_raw
            self.update_grow_prune_param(grow_param_raw, prune_param_raw)
        else:
            self.prod = prod
            self.col_val_tables = list()
            self.col_names = dataset.columns
            self.col_names = self.col_names[:-1]
            self.col_unique_values = list()
            self.grow_param_raw = grow_param_raw
            self.prune_param_raw = prune_param_raw
            self.create_tables_for_every_value(dataset)
            self.create_pos_neg_all_bitmaps(dataset)
            self.update_grow_prune_param(grow_param_raw, prune_param_raw)
            if len(dataset) != 0:
                for i in range(0, len(self.col_names)):
                    for j in range(0, len(self.col_unique_values[i])):
                        act_col = self.col_names[i]
                        act_value = self.col_unique_values[i][j]
                        self.col_val_tables[i].append(
                            BitmapDataset.create_bitmap_for_value(act_col, act_value, dataset))

    def create_pos_neg_all_bitmaps(self, dataset):
        pos_idx = dataset.index[dataset[dataset.columns[-1]] == 1].tolist()
        self.pos_map = BitMap(pos_idx)
        neg_idx = dataset.index[dataset[dataset.columns[-1]] == 0].tolist()
        self.neg_map = BitMap(neg_idx)
        self.all_id = self.pos_map | self.neg_map

    def update_grow_prune_param(self, grow_param_raw, prune_param_raw):
        if len(self.pos_map) == 0:
            self.grow_param = math.inf
            self.prune_param = math.inf
        else:
            self.grow_param = len(self.pos_map) * math.log(len(self.pos_map) / len(self.all_id), 2) * (
                -1) * grow_param_raw
            self.prune_param = len(self.pos_map) * math.log(len(self.pos_map) / len(self.all_id), 2) * (
                -1) * prune_param_raw

    def create_tables_for_every_value(self, dataset):
        for i in range(0, len(self.col_names)):
            self.col_unique_values.append(dataset[self.col_names[i]].unique())
            self.col_val_tables.append(list())

    @staticmethod
    def create_bitmap_for_value(act_col, act_value, dataset):
        return BitMap(dataset.index[dataset[act_col] == act_value])

    def delete_covered(self, rule):
        new_rule = self.make_rules_from_iters(rule)
        for i in range(0, len(self.col_val_tables)):
            for j in range(0, len(self.col_val_tables[i])):
                self.col_val_tables[i][j] = self.col_val_tables[i][j] - new_rule
        self.pos_map = self.pos_map - new_rule
        self.neg_map = self.neg_map - new_rule
        self.all_id = self.all_id - new_rule
        self.update_grow_prune_param(self.grow_param_raw, self.prune_param_raw)

    def delete_not_covered(self, rule):
        new_rule = self.make_rules_from_iters(rule)
        for i in range(0, len(self.col_val_tables)):
            for j in range(0, len(self.col_val_tables[i])):
                self.col_val_tables[i][j] = self.col_val_tables[i][j] & new_rule
        self.pos_map = self.pos_map & new_rule
        self.neg_map = self.neg_map & new_rule
        self.all_id = self.all_id & new_rule
        self.update_grow_prune_param(self.grow_param_raw, self.prune_param_raw)

    def grow_rule(self):

        return self.grow_rule_sorted_p_n()

        # return self.grow_rule_inductive()

    def grow_rule_sorted_p_n(self):
        best_rule = list()
        while True:
            best_l = None
            best_foil = -math.inf
            p0, n0 = self.count_p_n_rule(best_rule)
            for i in range(0, len(self.col_names)):
                if i not in [x[0] for x in best_rule]:
                    tmp_l, tmp_foil = self.find_best_literal_from_variable(i, p0, n0, best_rule)
                    if tmp_foil > best_foil:
                        best_l = copy.deepcopy(tmp_l)
                        best_foil = tmp_foil
            if best_foil > self.grow_param:
                best_rule = best_rule + best_l
            else:
                break
        return best_rule

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
            if best_foil > self.grow_param:
                best_rule.append(best_l)
            else:
                break
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
        if rule is None or len(rule) == 0:
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
        unique_atr = list(set([i[0] for i in rule]))
        len_rule_unique_atr = len(unique_atr)

        for i in range(len_rule_unique_atr - 1, -1, -1):
            pruned_rule = copy.deepcopy(rule)
            for j in range(len(rule) - 1, -1, -1):
                if rule[j][0] == unique_atr[i]:
                    del pruned_rule[j]
            p, n = self.count_p_n_rule(rule)
            p0, n0 = self.count_p_n_rule(pruned_rule)
            if count_foil_grow(p0, n0, p, n) <= self.prune_param:
                rule = copy.deepcopy(pruned_rule)
        print(p, n)
        if p == 0 or n >= p or len(rule) == 0:
            return None
        else:
            return rule

    def split_into_growset_pruneset(self):
        count_growset = round(self.length() * 2 / 3)
        idx = self.choose_idx_for_split(count_growset)
        ids_grow = BitMap()
        all_id_arr = self.all_id.to_array()
        for i in range(0, len(idx)):
            ids_grow.add(all_id_arr[idx[i]])
        col_val_tables_grow, col_val_tables_prune = self.split_by_idx(ids_grow)
        pos_map_grow = self.pos_map & ids_grow
        neg_map_grow = self.neg_map & ids_grow
        pos_map_prune = self.pos_map - ids_grow
        neg_map_prune = self.neg_map - ids_grow
        return BitmapDataset(prod=self.prod, col_val_tables=col_val_tables_grow,
                             col_names=self.col_names, col_unique_values=self.col_unique_values, pos_map=pos_map_grow,
                             neg_map=neg_map_grow, grow_param_raw=self.grow_param_raw,
                             prune_param_raw=self.prune_param_raw), \
               BitmapDataset(prod=self.prod, col_val_tables=col_val_tables_prune,
                             col_names=self.col_names, col_unique_values=self.col_unique_values, pos_map=pos_map_prune,
                             neg_map=neg_map_prune, grow_param_raw=self.grow_param_raw,
                             prune_param_raw=self.prune_param_raw)

    def choose_idx_for_split(self, count_growset):
        if self.prod == 1:
            idx = random.sample(range(0, self.length()), count_growset)
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
                col_val_tables_grow[i].append(self.col_val_tables[i][j] & idx)
                col_val_tables_prune[i].append(self.col_val_tables[i][j] - idx)
        return col_val_tables_grow, col_val_tables_prune

    def count_p_n_rule(self, rule):
        if len(rule) == 0:
            return len(self.pos_map), len(self.neg_map)
        else:
            new_rule = self.make_rules_from_iters(rule)
            p_rule = new_rule & self.pos_map
            n_rule = new_rule & self.neg_map
            return len(p_rule), len(n_rule)

    def make_rules_from_iters(self, rule):
        rule = sorted(rule, key=lambda x: x[0])
        act_rule = BitMap()
        act_rule_tmp = BitMap()
        prev_i = -1
        for i, j in rule:
            if prev_i != i:
                if len(act_rule) == 0:
                    act_rule = act_rule_tmp
                else:
                    act_rule = act_rule & act_rule_tmp
                act_rule_tmp = self.col_val_tables[i][j]
            else:
                act_rule_tmp = act_rule_tmp | self.col_val_tables[i][j]
            prev_i = i
        if len(act_rule) == 0:
            act_rule = act_rule_tmp
        else:
            act_rule = act_rule & act_rule_tmp
        return act_rule

    def is_any_pos_example(self):
        return len(self.pos_map) > 0

    def length(self):
        return len(self.all_id)
