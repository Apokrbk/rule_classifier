import pandas as pd
import math
import copy

from rule_induction_classifier.abstract_datasets.abstract_dataset import AbstractDataset, count_foil_grow
from rule_induction_classifier.literal import Literal
from rule_induction_classifier.rule import Rule


class DictDataset(AbstractDataset):
    def __init__(self, prod, dataset, grow_param_raw=0, prune_param_raw=0, roulette_selection=False):
        super().__init__(prod, dataset)
        self.df = dataset
        self.dict = dataset.to_dict()
        self.numeric_cols, self.char_cols = self.split_into_numeric_car_cols()
        self.class_name = self.df.columns[len(self.df.columns) - 1]
        self.prod = prod
        self.prune_param_raw = prune_param_raw
        self.grow_param_raw = grow_param_raw
        self.update_grow_prune_param()

    def update_grow_prune_param(self):
        pos = len(self.df[self.df[self.class_name] == 1])
        if pos == 0:
            self.grow_param = math.inf
            self.prune_param = math.inf
        else:
            self.grow_param = pos * math.log(pos / len(self.df), 2) * (
                -1) * self.grow_param_raw
            self.prune_param = pos * math.log(pos / len(self.df), 2) * (
                -1) * self.prune_param_raw

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
        self.update_grow_prune_param()

    def delete_not_covered(self, rule):
        idx = set()
        for i in range(0, len(self.df)):
            to_delete = True
            if len(rule.literals) == 0:
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
        self.update_grow_prune_param()

    def grow_rule(self):
        rule = Rule()
        growset = DictDataset(self.prod, self.df)
        while True:
            p0, n0 = growset.count_p_n_rule(rule)
            best_foil = -math.inf
            best_l = None
            for i in range(0, len(list(growset.dict.keys()))):
                if self.class_name!=list(growset.dict.keys())[i]:
                    col_name = list(growset.dict.keys())[i]
                    foil = best_foil
                    if col_name not in [x.var_name for x in rule.literals] or col_name in self.numeric_cols:
                        col_values = list(set(growset.dict[col_name].values()))
                        l, foil = growset.find_best_literal(p0, n0, col_values, col_name)
                    if foil > best_foil:
                        best_l = copy.deepcopy(l)
                        best_foil = foil
            if best_foil > self.grow_param:
                rule.add_literal(best_l)
                growset.delete_not_covered(rule)
            else:
                break
        return rule

    def find_best_literal(self, p0, n0, col_values, col_name):
        if col_name in self.numeric_cols:
            l, foil = self.find_best_num_literal(p0, n0, col_values, col_name)
        else:
            l, foil = self.find_best_char_literal(p0, n0, col_values, col_name)
        return l, foil

    def prune_rule(self, rule):
        not_pruned_rule = copy.deepcopy(rule)
        for i in range(len(rule.literals) - 1, -1, -1):
            pruned_rule = copy.deepcopy(rule)
            pruned_rule.delete_literal(not_pruned_rule.literals[i])
            p, n = self.count_p_n_rule(rule)
            p0, n0 = self.count_p_n_rule(pruned_rule)
            if count_foil_grow(p0, n0, p, n) <= self.prune_param:
                rule.delete_literal(not_pruned_rule.literals[i])
        p, n = self.count_p_n_rule(rule)
        if p == 0 or n >= p or len(rule.literals) == 0:
            return None
        else:
            return rule

    def split_into_growset_pruneset(self, ratio=2/3):
        if self.prod == 1:
            trainset = self.df.sample(frac=1)
        else:
            trainset = self.df
        trainset.index = range(len(trainset))
        div_idx = math.floor(len(trainset) * ratio)
        growset = trainset[0:div_idx]
        growset.index = range(len(growset))
        pruneset = trainset[div_idx:]
        pruneset.index = range(len(pruneset))
        return DictDataset(self.prod, growset, grow_param_raw=self.grow_param_raw,
                           prune_param_raw=self.prune_param_raw), \
               DictDataset(self.prod, pruneset, grow_param_raw=self.grow_param_raw,
                           prune_param_raw=self.prune_param_raw)

    def split_into_numeric_car_cols(self):
        numeric_cols = self.df._get_numeric_data().columns
        numeric_cols = numeric_cols[:-1]
        char_cols = self.df.select_dtypes('object').columns
        return numeric_cols, char_cols

    def make_rule(self, rule):
        for i in range(0, len(rule.literals)):
            try:
                rule.literals[i].values = sorted(rule.literals[i].values)
            except TypeError:
                pass
        return rule

    def unmake_rule(self, rule):
        return rule

    def count_p_n_rule(self, rule):
        p = 0
        n = 0
        if len(rule.literals) == 0:
            return self.empty_rule_count_p_n()
        return self.not_empty_rule_count_p_n(n, p, rule)

    def not_empty_rule_count_p_n(self, n, p, rule):
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

    def empty_rule_count_p_n(self):
        try:
            p = self.df[self.class_name].value_counts()[1]
        except KeyError:
            p = 0
        try:
            n = self.df[self.class_name].value_counts()[0]
        except KeyError:
            n = 0
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
            best_foil, best_l = self.check_literal(atr_col_name, best_foil, best_l, i, n0, p0, unique_values, '<')
            best_foil, best_l = self.check_literal(atr_col_name, best_foil, best_l, i, n0, p0, unique_values, '>')
        return best_l, best_foil

    def check_literal(self, atr_col_name, best_foil, best_l, i, n0, p0, unique_values, op):
        literal = Literal(atr_col_name, op, unique_values[i])
        p, n = self.count_p_n_literal(literal)
        tmp_foil = count_foil_grow(p0, n0, p, n)
        if tmp_foil > best_foil:
            best_foil = tmp_foil
            best_l = copy.deepcopy(literal)
        return best_foil, best_l

    def find_best_char_literal(self, p0, n0, unique_values, atr_col_name):
        best_foil = -math.inf
        best_l = None
        df = self.count_p_n_for_every_value_and_sort(atr_col_name, unique_values)
        values_to_literal = list()
        best_foil, best_l = self.choose_best_literal(atr_col_name, best_foil, best_l, df, n0, p0, unique_values,
                                                     values_to_literal)
        if best_foil == -math.inf:
            return None, best_foil
        else:
            return best_l, best_foil

    def choose_best_literal(self, atr_col_name, best_foil, best_l, df, n0, p0, unique_values, values_to_literal):
        for i in range(0, len(unique_values)):
            values_to_literal.append(df.at[i, 'value'])
            literal = Literal(atr_col_name, 'in', values_to_literal)
            p, n = self.count_p_n_literal(literal)
            foil = count_foil_grow(p0, n0, p, n)
            if foil > best_foil:
                best_foil = foil
                best_l = copy.deepcopy(literal)
            else:
                break
        return best_foil, best_l

    def count_p_n_for_every_value_and_sort(self, atr_col_name, unique_values):
        p_to_n = list()
        for i in range(0, len(unique_values)):
            literal = Literal(atr_col_name, 'in', unique_values[i])
            p, n = self.count_p_n_literal(literal)
            if n == 0:
                p_to_n.append(math.inf)
            else:
                p_to_n.append(p / n)
        df = pd.DataFrame({'value': unique_values, 'p_to_n': p_to_n})
        df = df.sort_values(by='p_to_n', ascending=False)
        df.index = range(len(df))
        return df

    def length(self):
        return len(self.df)
