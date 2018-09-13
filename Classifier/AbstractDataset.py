from abc import abstractmethod


class AbstractDataset:
    @abstractmethod
    def __init__(self, dataset, prod):
        pass

    @abstractmethod
    def delete_covered(self, rule):
        pass

    @abstractmethod
    def delete_not_covered(self, rule):
        pass

    @abstractmethod
    def grow_rule(self):
        pass

    @abstractmethod
    def make_rule(self):
        pass

    @abstractmethod
    def prune_rule(self, rule):
        pass

    @abstractmethod
    def split_into_growset_pruneset(self):
        pass

    @abstractmethod
    def count_p_n_rule(self, rule):
        pass

    @abstractmethod
    def is_any_pos_example(self):
        pass

    @abstractmethod
    def length(self):
        pass
