import math
import pandas as pd
import unittest
import numpy as np

from Classifier.abstract_datasets.nparray_dataset.nparray_dataset import NpArrayDataset, count_foil_grow
from Classifier.literal import Literal
from Classifier.rule import Rule


class TestNotebook(unittest.TestCase):
    # TEST CONSTRUCTOR
    def test_constructor_pos_map_1(self):
        df = pd.read_csv('test_files/testfile_6_all_n.csv',
                         encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(0, np.count_nonzero(ds.pos_map == True))
        self.assertEqual(32, len(ds.pos_map))

    def test_constructor_pos_map_2(self):
        df = pd.read_csv('test_files/testfile_10_all_p.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, np.count_nonzero(ds.pos_map == True))
        self.assertEqual(32, len(ds.pos_map))

    def test_constructor_pos_map_3(self):
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, len(ds.pos_map))
        idx = np.where(ds.pos_map == True)[0]
        self.assertEqual("[16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]", str(idx))

    def test_constructor_neg_map_1(self):
        df = pd.read_csv('test_files/testfile_6_all_n.csv',
                         encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, np.count_nonzero(ds.neg_map == True))
        self.assertEqual(32, len(ds.neg_map))

    def test_constructor_neg_map_2(self):
        df = pd.read_csv('test_files/testfile_10_all_p.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(0, np.count_nonzero(ds.neg_map == True))
        self.assertEqual(32, len(ds.neg_map))

    def test_constructor_neg_map_3(self):
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, len(ds.neg_map))
        idx = np.where(ds.neg_map == True)[0]
        self.assertEqual("[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]", str(idx))

    def test_constructor_number_of_columns_1(self):
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(3, len(ds.col_names))

    def test_constructor_number_of_unique_values_tables_1(self):
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(3, len(ds.col_unique_values))

    def test_constructor_number_of_unique_values_1(self):
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(4, len(ds.col_unique_values[0]))

    def test_constructor_number_of_unique_values_2(self):
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(2, len(ds.col_unique_values[1]))

    def test_constructor_number_of_unique_values_3(self):
        df = pd.read_csv('test_files/testfile_4.csv', encoding='utf-8',
                         delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(1, len(ds.col_unique_values[0]))

    # TEST CREATE BITMAP FOR VALUE
    def test_create_bitmap_for_value_1(self):
        df = pd.read_csv('test_files/testfile_3.csv', encoding='utf-8',
                         delimiter=';')
        bitmap = NpArrayDataset.create_bitmap_for_value('ClassOfSeat', '1st', df)
        idx = np.where(bitmap == True)[0]
        self.assertEqual(32, len(bitmap))
        self.assertEqual("[ 0  4  8 12 16 20 24 28]", str(idx))

    def test_create_bitmap_for_value_2(self):
        df = pd.read_csv('test_files/testfile_3.csv', encoding='utf-8',
                         delimiter=';')
        bitmap = NpArrayDataset.create_bitmap_for_value('Age', 5557, df)
        idx = np.where(bitmap == True)[0]
        self.assertEqual(32, len(bitmap))
        self.assertEqual("[12]", str(idx))

    def test_create_bitmap_for_value_3(self):
        df = pd.read_csv('test_files/testfile_4.csv', encoding='utf-8',
                         delimiter=';')
        bitmap = NpArrayDataset.create_bitmap_for_value('ClassOfSeat', '1st', df)
        idx = np.where(bitmap == True)[0]
        self.assertEqual(64, len(bitmap))
        self.assertEqual("[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n 24 25 26 27 28 "
                         "29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47\n 48 49 50 51 52 53 54 55 56 57 58"
                         " 59 60 61 62 63]", str(idx))

    # TEST IS ANY POS EXAMPLE
    def test_is_any_pos_example_true(self):
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(True, ds.is_any_pos_example())

    def test_is_any_pos_example_false(self):
        df = pd.read_csv('test_files/testfile_6_all_n.csv',
                         encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(False, ds.is_any_pos_example())

    def test_is_any_pos_example_true_all_p(self):
        df = pd.read_csv('test_files/testfile_10_all_p.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(True, ds.is_any_pos_example())

    # TEST COUNT_P_N RULES
    def test_rule_count_p_n1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n3(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(4, p)
        self.assertEqual(4, n)

    def test_rule_count_p_n4(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        rule = Rule()
        ds = NpArrayDataset(1, df)
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(16, p)
        self.assertEqual(16, n)

    def test_rule_count_p_n5(self):
        df = pd.read_csv('test_files/testfile_4.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('Sex', 'in', 'Female')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        ds.delete_covered(ds.unmake_rule(rule))
        p, n = ds.count_p_n_rule(ds.unmake_rule(rule))
        self.assertEqual(0, p)
        self.assertEqual(0, n)

    # TEST DELETE COVERED
    def test_delete_covered_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        rule = Rule()
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before, len_after)

    def test_delete_covered_2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
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
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 8, len_after)

    def test_delete_covered_4(self):
        df = pd.read_csv('test_files/testfile_4.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 64, len_after)

    # TEST DELETE NOT COVERED
    def test_delete_not_covered_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        rule = Rule()
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(0, len_after)

    def test_delete_not_covered_2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
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
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 24, len_after)

    def test_delete_not_covered_4(self):
        df = pd.read_csv('test_files/testfile_4.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('Sex', 'in', 'Male')
        rule = Rule()
        rule.add_literal(l)
        ds = NpArrayDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(ds.unmake_rule(rule))
        self.assertEqual(36, ds.length())
        ds.delete_not_covered(ds.unmake_rule(rule))
        len_after = ds.length()
        self.assertEqual(len_before - 64, len_after)

    # TEST SPLIT INTO GROWSET PRUNESET
    def test_split_into_growset_pruneset_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        grow, prune = ds.split_into_growset_pruneset()
        self.assertEqual(21, grow.length())
        self.assertEqual(11, prune.length())

    # TEST LENGTH
    def test_length_dataset_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, ds.length())

    def test_length_dataset_2_no_rows(self):
        df = pd.read_csv('test_files/testfile_11_no_rows.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(0, ds.length())

    def test_length_dataset_3_all_p(self):
        df = pd.read_csv('test_files/testfile_10_all_p.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, ds.length())

    def test_length_dataset_4_all_n(self):
        df = pd.read_csv('test_files/testfile_6_all_n.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(1, df)
        self.assertEqual(32, ds.length())

    # TEST PRUNE RULE
    def test_prune_rule_1(self):
        df = pd.read_csv('test_files/testfile_3.csv',
                         encoding='utf-8', delimiter=';')
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
        rule = ds.grow_rule_sorted_p_n()
        rule = ds.make_rule(rule)
        self.assertEqual("Sex in ['Female'] and Age in [4, 5, 6, 7, 35, 37, 45]", rule.to_string())

    def test_grow_rule_2(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = ds.grow_rule_sorted_p_n()
        rule = ds.make_rule(rule)
        self.assertEqual("a5 in ['c', 'f', 'm', 'p', 's', 'y']", rule.to_string())

    def test_grow_rule_3(self):
        df = pd.read_csv('test_files/testfile_4.csv',
                         encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = ds.grow_rule_inductive()
        rule = ds.make_rule(rule)
        self.assertEqual("Sex in ['Female'] and Age in [4, 5, 6, 7, 8, 9]", rule.to_string())

    def test_grow_rule_4(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = ds.grow_rule_inductive()
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

    #TEST FIND BEST LITERAL FROM VARIABLE
    def test_find_best_literal_from_variable_1(self):
        df = pd.read_csv('test_files/testfile_4.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        literal, foil = ds.find_best_literal_from_variable(1, 32,32, list())
        self.assertEqual("[[1, 0]]", str(literal))

    def test_find_best_literal_from_variable_2(self):
        df = pd.read_csv('test_files/testfile_4.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        literal, foil = ds.find_best_literal_from_variable(2, 32,32, list())
        self.assertEqual("[[2, 1], [2, 2], [2, 5], [2, 6], [2, 7], [2, 0], [2, 3]]", str(literal))

    def test_find_best_literal_from_variable_3(self):
        df = pd.read_csv('test_files/testfile_4.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        old_rule = list()
        old_rule.append([2,4])
        literal, foil = ds.find_best_literal_from_variable(1, 2,4, old_rule)
        self.assertEqual("[[1, 0]]", str(literal))

    #TEST MAKE RULES FROM ITERS
    def test_make_rules_from_iters_1(self):
        df = pd.read_csv('test_files/testfile_4.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = list()
        rule.append([0,0])
        bitmap = ds.make_rules_from_iters(rule)
        idx = np.where(bitmap == True)[0]
        self.assertEqual(64, len(bitmap))
        self.assertEqual("[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n 24 25 26 27 28 "
                         "29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47\n 48 49 50 51 52 53 54 55 56 57 58"
                         " 59 60 61 62 63]", str(idx))

    def test_make_rules_from_iters_2(self):
        df = pd.read_csv('test_files/testfile_4.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = list()
        rule.append([1, 1])
        bitmap = ds.make_rules_from_iters(rule)
        idx = np.where(bitmap == True)[0]
        self.assertEqual(64, len(bitmap))
        self.assertEqual("[ 8 10 11 16 17 18 19 21 24 25 26 27 28 30 40 42 43 48 49 50 51 53 56 57\n 58 59 60 62]", str(idx))

    def test_make_rules_from_iters_3(self):
        df = pd.read_csv('test_files/testfile_4.csv', encoding='utf-8', delimiter=';')
        ds = NpArrayDataset(0, df)
        rule = list()
        rule.append([1, 1])
        ds.delete_covered(rule)
        bitmap = ds.make_rules_from_iters(rule)
        idx = np.where(bitmap == True)[0]
        self.assertEqual(36, len(bitmap))
        self.assertEqual("[]", str(idx))