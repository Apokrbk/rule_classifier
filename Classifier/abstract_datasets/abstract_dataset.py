import math
from abc import abstractmethod


class AbstractDataset:
    @abstractmethod
    def __init__(self, prod, dataset):
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
    def make_rule(self, rule):
        pass

    @abstractmethod
    def unmake_rule(self, rule):
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
        # if n == 0 and n0 == 0:
        #     return p - p0
        try:
            return p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2))
        except (ZeroDivisionError, ValueError):
            return -math.inf