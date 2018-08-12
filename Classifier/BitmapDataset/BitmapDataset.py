import math

from Classifier.AbstractDataset import AbstractDataset
from pyroaring import BitMap
import copy
from random import shuffle

from Classifier.DictDataset.DictDataset import count_foil_grow
from Classifier.Literal import Literal
from Classifier.Rule import Rule


class BitmapDataset(AbstractDataset):
    def __init__(self, dataset=None, col_names=None, col_dicts=None, rows=None, uniq_value=None):
        super().__init__(dataset)
        if col_names is None:
            self.preprocess_data(dataset, 10)
            self.rows = list()
            self.col_names = dataset.columns
            self.col_names = self.col_names[:-1]
            self.col_dicts = list()
            it = 2
            for i in range(0, len(self.col_names)):
                unique_values = dataset[self.col_names[i]].unique()
                act_dict = dict()
                for j in range(0, len(unique_values)):
                    act_dict.update({unique_values[j]: it})
                    it += 1
                self.col_dicts.append(act_dict)
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
            self.uniq_val = it - 1
        else:
            self.col_names = col_names
            self.col_dicts = col_dicts
            self.rows = rows
            self.uniq_val = uniq_value

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
        while True:
            p0, n0 = self.count_p_n_rule(rule)
            best_foil = -math.inf
            for i in range(2, self.uniq_val):
                if i not in rule:
                    rule.add(i)
                    p, n = self.count_p_n_rule(rule)
                    foil = count_foil_grow(p0, n0, p, n)
                    if foil > best_foil:
                        l = i
                        best_foil = foil
                    rule.remove(i)
            if best_foil == 0 or best_foil == -math.inf:
                break
            rule.add(l)
        return rule

    def prune_rule(self, rule):
        literals = list(rule)
        for i in range(0, len(literals)):
            p, n = self.count_p_n_rule(rule)
            rule.remove(literals[i])
            p0, n0 = self.count_p_n_rule(rule)
            if p0 != 0 and p != 0:
                if p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2)) > 0:
                    rule.add(literals[i])
            if p != 0:
                rule.add(literals[i])
        p, n = self.count_p_n_rule(rule)
        if p == 0 or n >= p or len(rule) == 0:
            return None
        else:
            return rule

    def split_into_growset_pruneset(self):
        shuffle(self.rows)
        div_idx = math.floor(len(self.rows) * 2 / 3)
        return BitmapDataset(col_names=self.col_names, col_dicts=self.col_dicts, rows=self.rows[:div_idx],
                             uniq_value=self.uniq_val), BitmapDataset(
            col_names=self.col_names, col_dicts=self.col_dicts,
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

    def preprocess_data(self, df, num_of_intervals):
        numeric_cols = df._get_numeric_data().columns
        numeric_cols = numeric_cols[:-1]
        for i in range(0, len(numeric_cols)):
            min_value = df[numeric_cols[i]].min()
            max_value = df[numeric_cols[i]].max()
            interval = (max_value - min_value) / num_of_intervals
            start = min_value
            for j in range(0, num_of_intervals):
                if j == num_of_intervals-1:
                    df.loc[(df[numeric_cols[i]] >= start), numeric_cols[i]] = j
                else:
                    df.loc[(df[numeric_cols[i]] >= start) & (df[numeric_cols[i]] < start+interval), numeric_cols[i]] = j
                    start += interval
