import math
import time

import numpy as np

from Classifier.AbstractDataset import AbstractDataset
from pyroaring import BitMap
import copy
from random import shuffle

from Classifier.DictDataset.DictDataset import count_foil_grow
from Classifier.Literal import Literal
from Classifier.Rule import Rule


class BitmapDataset(AbstractDataset):
    def __init__(self, prod=1, dataset=None, col_names=None, col_dicts=None, rows=None, uniq_value=None):
        super().__init__(1, dataset)
        self.prod = prod
        if col_names is None:
            self.rows = list()
            self.col_names = dataset.columns
            self.col_names = self.col_names[:-1]
            self.col_dicts = list()
            it = self.create_dicts(dataset)
            self.create_rows(dataset)
            self.uniq_val = it - 1
        else:
            self.col_names = col_names
            self.col_dicts = col_dicts
            self.rows = rows
            self.uniq_val = uniq_value

    def create_rows(self, dataset):
        for j in range(0, len(dataset)):
            act_row = BitMap()
            for i in range(0, len(dataset.columns)):
                if i == len(dataset.columns) - 1:
                    if dataset[dataset.columns[i]][j] == 1:
                        act_row.add(1)
                    else:
                        act_row.add(0)
                else:
                    act_dict = self.col_dicts[i]
                    act_row.add(act_dict[dataset[self.col_names[i]][j]])
            self.rows.append(act_row)

    def create_dicts(self, dataset):
        it = 2
        for i in range(0, len(self.col_names)):
            unique_values = dataset[self.col_names[i]].unique()
            act_dict = dict()
            for j in range(0, len(unique_values)):
                act_dict.update({unique_values[j]: it})
                it += 1
            self.col_dicts.append(act_dict)
        return it

    def delete_covered(self, rule):
        self.rows = [x for x in self.rows if not self.row_covered_by_rule(x, rule)]

    def delete_not_covered(self, rule):
        self.rows = [x for x in self.rows if self.row_covered_by_rule(x, rule)]

    def row_covered_by_rule(self, row, rule):
        if len(rule) == 0:
            return False
        rule_len = len(rule)
        result = row & rule
        if len(result) == rule_len:
            return True
        else:
            return False

    def grow_rule(self):
        rule = BitMap()
        p0, n0 = self.count_p_n_rule(rule)
        best_foil = -math.inf
        l = self.find_first_literal(best_foil, n0, p0, rule)
        rule.add(l)
        self.check_other_literals(rule)
        return rule

    def check_other_literals(self, rule):
        for i in range(2, self.uniq_val):
            p, n = self.count_p_n_rule(rule)
            if i not in rule:
                rule.add(i)
                p0, n0 = self.count_p_n_rule(rule)
                if p0 != 0 and p != 0:
                    if n == 0 and n0 == 0:
                        if p >= p0:
                            rule.remove(i)
                    else:
                        if p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2)) > 0:
                            rule.remove(i)
                if p0 == 0:
                    rule.remove(i)

    def find_first_literal(self, best_foil, n0, p0, rule):
        for i in range(2, self.uniq_val):
            if i not in rule:
                rule.add(i)
                p, n = self.count_p_n_rule(rule)
                foil = count_foil_grow(p0, n0, p, n)
                if foil > best_foil:
                    l = i
                    best_foil = foil
                rule.remove(i)
        return l

    def prune_rule(self, rule):
        literals = list(rule)
        for i in range(0, len(literals)):
            p, n = self.count_p_n_rule(rule)
            rule.remove(literals[i])
            p0, n0 = self.count_p_n_rule(rule)
            if p0 != 0 and p != 0:
                if p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2)) > 0:
                    rule.add(literals[i])
            if p0 == 0:
                rule.add(literals[i])
        p, n = self.count_p_n_rule(rule)
        if p == 0 or n >= p or len(rule) == 0:
            return None
        else:
            return rule

    def split_into_growset_pruneset(self):
        if self.prod == 1:
            shuffle(self.rows)
        div_idx = math.floor(len(self.rows) * 2 / 3)
        return BitmapDataset(prod=self.prod, col_names=self.col_names, col_dicts=self.col_dicts,
                             rows=self.rows[:div_idx],
                             uniq_value=self.uniq_val), BitmapDataset(
            prod=self.prod, col_names=self.col_names, col_dicts=self.col_dicts,
            rows=self.rows[div_idx:], uniq_value=self.uniq_val)

    def count_p_n_rule(self, rule):
        if len(rule) == 0:
            return 0, 0
        p_rule = copy.deepcopy(rule)
        n_rule = copy.deepcopy(rule)
        p_rule.add(1)
        n_rule.add(0)
        p = 0
        n = 0
        for x in self.rows:
            if self.row_covered_by_rule(x, p_rule):
                p += 1
            elif self.row_covered_by_rule(x, n_rule):
                n += 1
        return p, n

    def is_any_pos_example(self):
        pos = BitMap()
        pos.add(1)
        for x in self.rows:
            if self.row_covered_by_rule(x, pos):
                return True
        return False

    def length(self):
        return len(self.rows)

    def make_rule(self, rule):
        literals = list(rule)
        new_rule = Rule()
        for i in range(0, len(literals)):
            for j in range(0, len(self.col_dicts)):
                if literals[i] in list(self.col_dicts[j].values()):
                    l = Literal(self.col_names[j], 'in',
                                list(self.col_dicts[j].keys())[list(self.col_dicts[j].values()).index(literals[i])])
                    new_rule.add_literal(l)
        return new_rule

    def unmake_rule(self, rule):
        new_rule = BitMap()
        for i in range(0, len(rule.literals)):
            for j in range(0, len(self.col_names)):
                if rule.literals[i].var_name == self.col_names[j]:
                    col = j
                    break
            for value in list(self.col_dicts[col].keys()):
                if rule.literals[i].value_covered_by_literal(value):
                    new_rule.add(self.col_dicts[col][value])
        return new_rule

