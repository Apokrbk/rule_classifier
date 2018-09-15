import math
import pandas as pd
import unittest

from Classifier.NpArrayDataset.NpArrayDataset import count_foil_grow, NpArrayDataset
from Classifier.Literal import Literal
from Classifier.NpArrayDataset.NpArrayDataset import NpArrayDataset
from Classifier.Rule import Rule


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

    # TEST IS ANY POS EXAMPLE
    def test_is_any_pos_example_true(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_is_any_pos_true.csv', encoding='utf-8',
                         delimiter=',')
        ds = NpArrayDataset(1,df)
        self.assertEqual(True, ds.is_any_pos_example())

    def test_is_any_pos_example_false(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_is_any_pos_false.csv',
                         encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1,df)
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


    # TEST COUNT_P_N RULES
    def test_rule_count_p_n1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1,df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n2(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1,df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n3(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1,df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n4(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        rule = Rule()
        ds = NpArrayDataset(1,df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(0, p)
        self.assertEqual(0, n)

    #TEST DELETE COVERED
    def test_delete_covered_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        rule = Rule()
        ds = NpArrayDataset(1,df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before, len_after)

    def test_delete_covered_2(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1,df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before-8, len_after)

    def test_delete_covered_3(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1,df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before-8, len_after)

    #TEST DELETE NOT COVERED
    def test_delete_not_covered_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        rule = Rule()
        ds = NpArrayDataset(1,df)
        len_before = ds.length()
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(0, len_after)

    def test_delete_not_covered_2(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1,df)
        len_before = ds.length()
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 24, len_after)

    def test_delete_not_covered_3(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1,df)
        len_before = ds.length()
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 24, len_after)

    #TEST SPLIT INTO GROWSET PRUNESET
    def test_split_into_growset_pruneset_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1,df)
        grow, prune = ds.split_into_growset_pruneset()
        self.assertEqual(22, grow.length())
        self.assertEqual(10, prune.length())

    #TEST LENGTH
    def test_length_dataset_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1,df)
        self.assertEqual(32, ds.length())

    #TEST PRUNE RULE
    def test_prune_rule_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_find_best_num_3.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('Sex', 'in', 'Female')
        l2 = Literal('Sex', 'in', 'Male')
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        ds = NpArrayDataset(1,df)
        rule = ds.prune_rule(ds.unmake_rule(rule))
        rule = ds.make_rule(rule)
        self.assertEqual("Sex in ['Female'] ", rule.to_string())

    #TEST GROW RULE
    def test_grow_rule_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_find_best_num_4.csv',
                         encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0,df)
        rule = ds.grow_rule()
        rule = ds.make_rule(rule)
        self.assertEqual("Sex in ['Female'] and Age in [7, 4, 6, 5, 9, 8] ", rule.to_string())


