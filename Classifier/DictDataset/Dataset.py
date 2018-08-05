import pandas as pd
import math
import copy

from Classifier.AbstractDataset import AbstractDataset
from Classifier.Literal import Literal
from Classifier.Rule import Rule


class Dataset(AbstractDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.df = dataset
        self.dict = dataset.to_dict()
        self.numeric_cols, self.char_cols = self.split_into_numeric_car_cols()
        self.class_name = self.df.columns[len(self.df.columns) - 1]

    def delete_covered(self, rule):
        idx = set()
        if len(rule.literals) == 0:
            return
        for i in range(0, len(self.df)):
            to_delete = True
            for j in range(0, len(rule.literals)):
                if not rule.literals[j].value_covered_by_literal(self.dict[rule.literals[j].var_name][i]):
                    to_delete = False
                    break
            if to_delete:
                idx.add(i)
        indexes_to_keep = set(range(self.df.shape[0])) - idx
        self.df = self.df.take(list(indexes_to_keep))
        self.df.index = range(len(self.df))
        self.dict = self.df.to_dict()

    def delete_not_covered(self, rule):
        idx = set()
        for i in range(0, len(self.df)):
            to_delete = True
            if len(rule.literals)==0:
                to_delete = False
            for j in range(0, len(rule.literals)):
                if not rule.literals[j].value_covered_by_literal(self.dict[rule.literals[j].var_name][i]):
                    to_delete = False
                    break
            if not to_delete:
                idx.add(i)
        indexes_to_keep = set(range(self.df.shape[0])) - idx
        self.df = self.df.take(list(indexes_to_keep))
        self.df.index = range(len(self.df))
        self.dict = self.df.to_dict()

    def grow_rule(self):
        rule = Rule()
        growset = Dataset(self.df)
        while True:
            p0, n0 = growset.count_p_n_rule(rule)
            best_foil = -math.inf
            for i in range(0, len(list(growset.dict.keys())) - 1):
                col_name = list(growset.dict.keys())[i]
                col_values = list(set(growset.dict[col_name].values()))
                l,foil = growset.find_best_literal(p0,n0,col_values, col_name)
                if foil > best_foil:
                    best_l = copy.deepcopy(l)
                    best_foil = foil
            if best_foil == 0 or best_foil == -math.inf:
                break
            rule.add_literal(best_l)
            growset.delete_not_covered(rule)
        return rule

    def find_best_literal(self,p0,n0,col_values,col_name):
        if col_name in self.numeric_cols:
            l, foil = self.find_best_num_literal(p0, n0, col_values, col_name)
        else:
            l, foil = self.find_best_char_literal(p0, n0, col_values, col_name)
        return l,foil

    def prune_rule(self, rule):
        not_pruned_rule = copy.deepcopy(rule)
        for i in range(len(rule.literals) - 1, -1, -1):
            pruned_rule = copy.deepcopy(rule)
            pruned_rule.delete_literal(not_pruned_rule.literals[i])
            p, n = self.count_p_n_rule(rule)
            p0, n0 = self.count_p_n_rule(pruned_rule)
            if p0 != 0 and p != 0:
                if p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2)) <= 0:
                    rule.delete_literal(not_pruned_rule.literals[i])
            if p == 0:
                rule.delete_literal(not_pruned_rule.literals[i])
        return rule

    def split_into_growset_pruneset(self):
        trainset = self.df.sample(frac=1)
        trainset.index = range(len(trainset))
        div_idx = math.floor(len(trainset) * 2 / 3)
        growset = trainset[0:div_idx]
        growset.index = range(len(growset))
        pruneset = trainset[div_idx:]
        pruneset.index = range(len(pruneset))
        return Dataset(growset), Dataset(pruneset)

    def split_into_numeric_car_cols(self):
        numeric_cols = self.df._get_numeric_data().columns
        numeric_cols = numeric_cols[:-1]
        char_cols = self.df.select_dtypes('object').columns
        return numeric_cols, char_cols

    def count_p_n_rule(self, rule):
        p = 0
        n = 0
        if len(rule.literals) == 0:
            return 0, 0
        for i in range(0, len(self.dict[self.class_name])):
            covered = True
            for j in range(0, len(rule.literals)):
                if not rule.literals[j].value_covered_by_literal(self.dict[rule.literals[j].var_name][i]):
                    covered = False
                    break
            if covered:
                if self.dict[self.class_name][i] == 1:
                    p += 1
                else:
                    n += 1
        return p, n

    def count_p_n_literal(self, literal):
        p = 0
        n = 0
        for i in range(0, len(self.dict[self.class_name])):
            if literal.value_covered_by_literal(self.dict[literal.var_name][i]):
                if self.dict[self.class_name][i] == 1:
                    p += 1
                else:
                    n += 1
        return p, n

    def is_any_pos_example(self):
        is_any = False
        for i in range(0, len(self.dict[self.class_name])):
            if self.dict[self.class_name][i] == 1:
                is_any = True
                break
        return is_any

    def find_best_num_literal(self, p0, n0, unique_values, atr_col_name):
        best_foil = -math.inf
        best_l = None
        for i in range(0, len(unique_values)):
            l = Literal(atr_col_name, '<', unique_values[i])
            p, n = self.count_p_n_literal(l)
            tmp_foil = count_foil_grow(p0, n0, p, n)
            if tmp_foil > best_foil:
                best_foil = tmp_foil
                best_l = copy.deepcopy(l)
            l = Literal(atr_col_name, '>', unique_values[i])
            p, n = self.count_p_n_literal(l)
            tmp_foil = count_foil_grow(p0, n0, p, n)
            if tmp_foil > best_foil:
                best_foil = tmp_foil
                best_l = copy.deepcopy(l)
        return best_l, best_foil

    def find_best_char_literal(self, p0, n0, unique_values, atr_col_name):
        best_foil = -math.inf
        p_to_n = list()
        for i in range(0, len(unique_values)):
            l = Literal(atr_col_name, 'in', unique_values[i])
            p, n = self.count_p_n_literal(l)
            if n == 0:
                p_to_n.append(math.inf)
            else:
                p_to_n.append(p / n)
        df = pd.DataFrame({'value': unique_values, 'p_to_n': p_to_n})
        df = df.sort_values(by='p_to_n', ascending=False)
        values_to_literal = list()
        for i in range(0, len(unique_values)):
            values_to_literal.append(df.at[i, 'value'])
            l = Literal(atr_col_name, 'in', values_to_literal)
            p, n = self.count_p_n_literal(l)
            foil = count_foil_grow(p0, n0, p, n)
            if foil > best_foil:
                best_foil = foil
                best_l = copy.deepcopy(l)
        if best_foil == -math.inf:
            return None, best_foil
        else:
            return best_l, best_foil

    def length(self):
        return len(self.df)


def count_foil_grow(p0, n0, p, n):
    if p0 == 0 and n0 == 0:
        if p == 0:
            return -math.inf
        try:
            return p * (p/(p+n))
        except (ZeroDivisionError, ValueError):
            return -math.inf
    else:
        if p == 0:
            return -math.inf
        try:
            return p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2))
        except (ZeroDivisionError, ValueError):
            return -math.inf