import math
import pandas as pd
import unittest

from Classifier.DictDataset import count_foil_grow, DictDataset
from Classifier.Literal import Literal
from Classifier.Rule import Rule


class TestNotebook(unittest.TestCase):

    # TEST FOIL GROW
    def test_count_foil_grow_all_zeros(self):
        self.assertEqual(-math.inf, count_foil_grow(0, 0, 0, 0))

    def test_count_foil_grow_p0_and_n0_zeros(self):
        self.assertEqual(round(4.6797, 4), round(count_foil_grow(0, 0, 8, 4), 4))

    def test_count_foil_grow_p_zero(self):
        self.assertEqual(-math.inf, count_foil_grow(6, 4, 0, 2))

    def test_count_foil_grow_all_not_zero(self):
        self.assertEqual(round(1.7531, 4), round(count_foil_grow(7, 3, 6, 1), 4))

    # TEST IS ANY POS EXAMPLE
    def test_is_any_pos_example_true(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_is_any_pos_true.csv', encoding='utf-8',
                         delimiter=',')
        ds = DictDataset(df)
        self.assertEqual(True, ds.is_any_pos_example())

    def test_is_any_pos_example_false(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_is_any_pos_false.csv',
                         encoding='utf-8', delimiter=',')
        ds = DictDataset(df)
        self.assertEqual(False, ds.is_any_pos_example())

    # TEST LITERALS
    def test_value_covered_by_literal_less_true(self):
        l = Literal('test', '<', 15)
        self.assertEqual(True, l.value_covered_by_literal(14))

    def test_value_covered_by_literal_less_false(self):
        l = Literal('test', '<', 15)
        self.assertEqual(False, l.value_covered_by_literal(15))

    def test_value_covered_by_literal_more_true(self):
        l = Literal('test', '>', -15)
        self.assertEqual(True, l.value_covered_by_literal(-14))

    def test_value_covered_by_literal_more_false(self):
        l = Literal('test', '>', -15)
        self.assertEqual(False, l.value_covered_by_literal(-15))

    def test_value_covered_by_literal_in_true(self):
        l = Literal('test', 'in', ['1st', '2nd'])
        self.assertEqual(True, l.value_covered_by_literal('1st'))

    def test_value_covered_by_literal_in_false(self):
        l = Literal('test', 'in', ['1st', '2nd'])
        self.assertEqual(False, l.value_covered_by_literal('3rd'))

    # TEST DATASET COUNT_P_N LITERAL
    def test_literal_count_p_n1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', ['1st', '2nd'])
        ds = DictDataset(df)
        p, n = ds.count_p_n_literal(l)
        self.assertEqual(8, p)
        self.assertEqual(8, n)

    def test_literal_count_p_n2(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', ['xab', 'asdas'])
        ds = DictDataset(df)
        p, n = ds.count_p_n_literal(l)
        self.assertEqual(0, p)
        self.assertEqual(0, n)

    def test_literal_count_p_n3(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('Age', '>', 1000)
        ds = DictDataset(df)
        p, n = ds.count_p_n_literal(l)
        self.assertEqual(0, p)
        self.assertEqual(4, n)

    def test_literal_count_p_n4(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('Sex', 'in', 'Male')
        ds = DictDataset(df)
        p, n = ds.count_p_n_literal(l)
        self.assertEqual(8, p)
        self.assertEqual(8, n)

    # TEST COUNT_P_N RULES
    def test_rule_count_p_n1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        l2 = Literal('Age', '<', 20)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        ds = DictDataset(df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(1, p)
        self.assertEqual(1, n)

    def test_rule_count_p_n2(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        l2 = Literal('Age', '<', 100)
        l3 = Literal('Age', '>', 15)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        rule.add_literal(l3)
        ds = DictDataset(df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(4, p)
        self.assertEqual(0, n)

    def test_rule_count_p_n3(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        l2 = Literal('Age', '<', 20)
        l3 = Literal('Age', '>', 30)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        rule.add_literal(l3)
        ds = DictDataset(df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(0, p)
        self.assertEqual(0, n)

    def test_rule_count_p_n4(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        rule = Rule()
        ds = DictDataset(df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(0, p)
        self.assertEqual(0, n)
