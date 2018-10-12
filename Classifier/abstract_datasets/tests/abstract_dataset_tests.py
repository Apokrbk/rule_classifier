import math
import pandas as pd
import unittest

from Classifier.abstract_datasets.abstract_dataset import count_foil_grow


class TestNotebook(unittest.TestCase):

    # TEST FOIL GROW
    def test_count_foil_grow_all_zeros(self):
        self.assertEqual(-math.inf, count_foil_grow(0, 0, 0, 0))

    def test_count_foil_grow_p0_and_n0_zeros(self):
        self.assertEqual(round(5.3333, 4), round(count_foil_grow(0, 0, 8, 4), 4))

    def test_count_foil_grow_p_zero(self):
        self.assertEqual(-math.inf, count_foil_grow(6, 4, 0, 2))

    def test_count_foil_grow_all_not_zero(self):
        self.assertEqual(round(1.7531, 4), round(count_foil_grow(7, 3, 6, 1), 4))
