import pandas as pd

from Classifier.abstract_datasets.bitmap_dataset.bitmap_dataset import BitmapDataset


class RuleCreator:

    def __init__(self, dataset_type=BitmapDataset, prod=1, grow_param_raw=0, prune_param_raw=0):
        self.prod = prod
        self.rules = list()
        self.dataset_type = dataset_type
        self.grow_param_raw = grow_param_raw
        self.prune_param_raw = prune_param_raw

    def fit(self, df_x, df_y):
        df_x['__class__'] = df_y
        trainset = self.dataset_type(self.prod, df_x, grow_param_raw=self.grow_param_raw,
                                     prune_param_raw=self.prune_param_raw)
        rules = list()
        max_iter = 0
        while max_iter < 5 and trainset.is_any_pos_example():
            growset, pruneset = trainset.split_into_growset_pruneset()
            new_rule = growset.grow_rule()
            new_rule = pruneset.prune_rule(new_rule)
            if new_rule is None:
                max_iter += 1
            else:
                trainset.delete_covered(new_rule)
                new_rule = trainset.make_rule(new_rule)
                rules.append(new_rule)
        self.rules = rules

    def predict(self, dataset):
        predictions = list()
        for index, row in dataset.iterrows():
            if self.row_covered(row):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def row_covered(self, row):
        if len(self.rules) == 0:
            return True
        for i in range(0, len(self.rules)):
            if self.rules[i].row_covered(row):
                return True
        return False

    def get_rules(self):
        return self.rules

    def get_number_of_rules(self):
        return len(self.rules)

    def print_rules(self):
        for i in range(0, len(self.rules)):
            print(self.rules[i].to_string())
