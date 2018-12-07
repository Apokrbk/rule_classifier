class RuleCreator:

    def __init__(self, dataset_type, df, prod):
        self.prod = prod
        self.rules = self.train(dataset_type(prod, df))

    def train(self, trainset):
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
        return rules

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




