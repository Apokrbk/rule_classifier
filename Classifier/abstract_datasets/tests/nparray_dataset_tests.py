import math
import pandas as pd
import unittest

from Classifier.abstract_datasets.nparray_dataset.nparray_dataset import NpArrayDataset, count_foil_grow
from Classifier.literal import Literal
from Classifier.rule import Rule


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
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=',')
        ds = NpArrayDataset(1, df)
        self.assertEqual(True, ds.is_any_pos_example())

    def test_is_any_pos_example_false(self):
        df = pd.read_csv('test_files/testfile_6_all_n.csv',
                         encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1, df)
        self.assertEqual(False, ds.is_any_pos_example())

    def test_is_any_pos_example_true_all_p(self):
        df = pd.read_csv('test_files/testfile_10_all_p.csv', encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1, df)
        self.assertEqual(True, ds.is_any_pos_example())

    # TEST COUNT_P_N RULES
    def test_rule_count_p_n1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n3(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n4(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        rule = Rule()
        ds = NpArrayDataset(1, df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(16, p)
        self.assertEqual(16, n)

    # TEST DELETE COVERED
    def test_delete_covered_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        rule = Rule()
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before, len_after)

    def test_delete_covered_2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 8, len_after)

    def test_delete_covered_3(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 8, len_after)

    # TEST DELETE NOT COVERED
    def test_delete_not_covered_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        rule = Rule()
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(0, len_after)

    def test_delete_not_covered_2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 24, len_after)

    def test_delete_not_covered_3(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 24, len_after)

    # TEST SPLIT INTO GROWSET PRUNESET
    def test_split_into_growset_pruneset_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1, df)
        grow, prune = ds.split_into_growset_pruneset()
        self.assertEqual(22, grow.length())
        self.assertEqual(10, prune.length())

    # TEST LENGTH
    def test_length_dataset_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, ds.length())

    def test_length_dataset_2_no_rows(self):
        df = pd.read_csv('test_files/testfile_11_no_rows.csv', encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1, df)
        self.assertEqual(0, ds.length())

    def test_length_dataset_3_all_p(self):
        df = pd.read_csv('test_files/testfile_10_all_p.csv', encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, ds.length())

    def test_length_dataset_4_all_n(self):
        df = pd.read_csv('test_files/testfile_6_all_n.csv', encoding='utf-8', delimiter=',')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, ds.length())

    # TEST PRUNE RULE
    def test_prune_rule_1(self):
        df = pd.read_csv('test_files/testfile_3.csv',
                         encoding='utf-8', delimiter=',')
        l = Literal('Sex', 'in', 'Female')
        l2 = Literal('Sex', 'in', 'Male')
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        ds = NpArrayDataset(1, df)
        rule = ds.prune_rule(ds.unmake_rule(rule))
        rule = ds.make_rule(rule)
        self.assertEqual("Sex in ['Female']", rule.to_string())

    def test_prune_rule_2(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        l = Literal('a5', 'in', ['c', 'f', 'm', 'p', 's', 'y'])
        rule = Rule()
        rule.add_literal(l)
        rule = ds.unmake_rule(rule)
        rule = ds.prune_rule(rule)
        rule = ds.make_rule(rule)
        self.assertEqual("a5 in ['c', 'f', 'm', 'p', 's', 'y']", rule.to_string())

    def test_prune_rule_3(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        l = Literal('a5', 'in', ['c', 'f', 'm', 'p', 's', 'y'])
        l2 = Literal('a3', 'in', 'e')
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        rule = ds.unmake_rule(rule)
        rule = ds.prune_rule(rule)
        rule = ds.make_rule(rule)
        self.assertEqual("a5 in ['c', 'f', 'm', 'p', 's', 'y']", rule.to_string())

    # TEST GROW RULE
    def test_grow_rule_1(self):
        df = pd.read_csv('test_files/testfile_4.csv',
                         encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = ds.grow_rule()
        rule = ds.make_rule(rule)
        self.assertEqual("Sex in ['Female'] and Age in [4, 5, 6, 7, 35, 37, 45]", rule.to_string())

    def test_grow_rule_2(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = ds.grow_rule()
        rule = ds.make_rule(rule)
        self.assertEqual("a5 in ['c', 'f', 'm', 'p', 's', 'y']", rule.to_string())

    # TEST MAKE RULE
    def test_make_rule_1(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = list()
        rule.append([0, 0])
        rule.append([2, 3])
        rule.append([5, 1])
        rule.append([5, 0])
        rule = ds.make_rule(rule)
        self.assertEqual("a1 in ['x'] and a3 in ['g'] and a6 in ['a', 'f']", rule.to_string())

    def test_make_rule_2(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = list()
        rule = ds.make_rule(rule)
        self.assertEqual("", rule.to_string())

    # TEST UNMAKE RULE
    def test_unmake_rule_1(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = Rule()
        l = Literal('a1', 'in', 'x')
        l2 = Literal('a3', 'in', 'g')
        l3 = Literal('a6', 'in', ['a', 'f'])
        rule.add_literal(l)
        rule.add_literal(l2)
        rule.add_literal(l3)
        rule = ds.unmake_rule(rule)
        self.assertEqual("[[0, 0], [2, 3], [5, 0], [5, 1]]", str(rule))

    def test_unmake_rule_2(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = Rule()
        rule = ds.unmake_rule(rule)
        self.assertEqual("[]", str(rule))
