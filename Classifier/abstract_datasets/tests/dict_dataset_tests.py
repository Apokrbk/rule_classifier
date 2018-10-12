import math
import pandas as pd
import unittest

from Classifier.abstract_datasets.dict_dataset.dict_dataset import DictDataset, count_foil_grow
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
    def test_is_any_pos_example_true_some_p_some_n(self):
        df = pd.read_csv('test_files/testfile_7_not_all_n.csv', encoding='utf-8',
                         delimiter=';')
        ds = DictDataset(1, df)
        self.assertEqual(True, ds.is_any_pos_example())

    def test_is_any_pos_example_false_all_n(self):
        df = pd.read_csv('test_files/testfile_6_all_n.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        self.assertEqual(False, ds.is_any_pos_example())

    def test_is_any_pos_example_true_all_p(self):
        df = pd.read_csv('test_files/testfile_10_all_p.csv', encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        self.assertEqual(True, ds.is_any_pos_example())

    # TEST DATASET COUNT_P_N LITERAL
    def test_literal_count_p_n1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', ['1st', '2nd'])
        ds = DictDataset(1, df)
        p, n = ds.count_p_n_literal(l)
        self.assertEqual(8, p)
        self.assertEqual(8, n)

    def test_literal_count_p_n2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', ['xab', 'asdas'])
        ds = DictDataset(1, df)
        p, n = ds.count_p_n_literal(l)
        self.assertEqual(0, p)
        self.assertEqual(0, n)

    def test_literal_count_p_n3(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('Age', '>', 1000)
        ds = DictDataset(1, df)
        p, n = ds.count_p_n_literal(l)
        self.assertEqual(0, p)
        self.assertEqual(4, n)

    def test_literal_count_p_n4(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('Sex', 'in', 'Male')
        ds = DictDataset(1, df)
        p, n = ds.count_p_n_literal(l)
        self.assertEqual(8, p)
        self.assertEqual(8, n)



    # TEST COUNT_P_N RULES
    def test_rule_count_p_n1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        l2 = Literal('Age', '<', 20)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        ds = DictDataset(1, df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(1, p)
        self.assertEqual(1, n)

    def test_rule_count_p_n2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        l2 = Literal('Age', '<', 100)
        l3 = Literal('Age', '>', 15)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        rule.add_literal(l3)
        ds = DictDataset(1, df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(4, p)
        self.assertEqual(0, n)

    def test_rule_count_p_n3(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        l2 = Literal('Age', '<', 20)
        l3 = Literal('Age', '>', 30)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        rule.add_literal(l3)
        ds = DictDataset(1, df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(0, p)
        self.assertEqual(0, n)

    def test_rule_count_p_n4(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        rule = Rule()
        ds = DictDataset(1, df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(16, p)
        self.assertEqual(16, n)

    # TEST DELETE COVERED
    def test_delete_covered_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        rule = Rule()
        ds = DictDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before, len_after)

    def test_delete_covered_2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        l2 = Literal('Age', '<', 100)
        l3 = Literal('Age', '>', 15)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        rule.add_literal(l3)
        ds = DictDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before - 4, len_after)

    def test_delete_covered_3(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        l2 = Literal('Age', '<', 20)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        ds = DictDataset(1, df)
        len_before = ds.length()
        ds.delete_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before - 2, len_after)

    # TEST DELETE NOT COVERED
    def test_delete_not_covered_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        rule = Rule()
        ds = DictDataset(1, df)
        len_before = ds.length()
        ds.delete_not_covered(rule)
        len_after = ds.length()
        self.assertEqual(0, len_after)

    def test_delete_not_covered_2(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', 'Crew')
        l2 = Literal('Age', '<', 100)
        l3 = Literal('Age', '>', 15)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        rule.add_literal(l3)
        ds = DictDataset(1, df)
        len_before = ds.length()
        ds.delete_not_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before - 28, len_after)

    def test_delete_not_covered_3(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('ClassOfSeat', 'in', '1st')
        l2 = Literal('Age', '<', 20)
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        ds = DictDataset(1, df)
        len_before = ds.length()
        ds.delete_not_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before - 30, len_after)

    # TEST SPLIT INTO GROWSET PRUNESET
    def test_split_into_growset_pruneset_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        grow, prune = ds.split_into_growset_pruneset()
        self.assertEqual(21, grow.length())
        self.assertEqual(11, prune.length())

    # TEST LENGTH
    def test_length_dataset_1(self):
        df = pd.read_csv('test_files/testfile_8.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        self.assertEqual(32, ds.length())

    def test_length_dataset_2_no_rows(self):
        df = pd.read_csv('test_files/testfile_11_no_rows.csv', encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        self.assertEqual(0, ds.length())

    def test_length_dataset_3_all_p(self):
        df = pd.read_csv('test_files/testfile_10_all_p.csv', encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        self.assertEqual(32, ds.length())

    def test_length_dataset_4_all_n(self):
        df = pd.read_csv('test_files/testfile_6_all_n.csv', encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        self.assertEqual(32, ds.length())

    # TEST FIND BEST NUM LITERAL
    def test_find_best_num_literal_1(self):
        df = pd.read_csv('test_files/testfile_1.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        best_l, best_foil = ds.find_best_literal(0, 0, df['Age'].unique(), 'Age')
        self.assertEqual(11.63636, round(best_foil, 5))
        self.assertEqual('Age < 455', best_l.to_string())

    def test_find_best_num_literal_2(self):
        df = pd.read_csv('test_files/testfile_2.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        best_l, best_foil = ds.find_best_literal(0, 0, df['Age'].unique(), 'Age')
        self.assertEqual(16, round(best_foil, 5))
        self.assertEqual('Age < 95', best_l.to_string())

    def test_find_best_num_literal_3(self):
        df = pd.read_csv('test_files/testfile_3.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        best_l, best_foil = ds.find_best_literal(0, 0, df['Age'].unique(), 'Age')
        self.assertEqual(10, round(best_foil, 5))
        self.assertEqual('Age > 88', best_l.to_string())

    # TEST FIND BEST CHAR LITERAL
    def test_find_best_char_literal_1(self):
        df = pd.read_csv('test_files/testfile_1.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        best_l, best_foil = ds.find_best_literal(0, 0, df['Sex'].unique(), 'Sex')
        self.assertEqual(8, round(best_foil, 5))
        self.assertEqual("Sex in ['Male', 'Female']", best_l.to_string())

    def test_find_best_char_literal_2(self):
        df = pd.read_csv('test_files/testfile_2.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        best_l, best_foil = ds.find_best_literal(0, 0, df['ClassOfSeat'].unique(), 'ClassOfSeat')
        self.assertEqual(9, round(best_foil, 5))
        self.assertEqual("ClassOfSeat in ['1st', '3rd', 'Crew']", best_l.to_string())

    def test_find_best_char_literal_3(self):
        df = pd.read_csv('test_files/testfile_3.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(1, df)
        best_l, best_foil = ds.find_best_literal(0, 0, df['Sex'].unique(), 'Sex')
        self.assertEqual(9.38889, round(best_foil, 5))
        self.assertEqual("Sex in ['Female']", best_l.to_string())

    # TEST PRUNE RULE
    def test_prune_rule_1(self):
        df = pd.read_csv('test_files/testfile_3.csv',
                         encoding='utf-8', delimiter=';')
        l = Literal('Sex', 'in', 'Female')
        l2 = Literal('Sex', 'in', 'Male')
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        ds = DictDataset(1, df)
        rule = ds.prune_rule(rule)
        self.assertEqual("Sex in Female", rule.to_string())

    def test_prune_rule_2(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = DictDataset(0, df)
        l = Literal('a5', 'in', ['c', 'f', 'm', 'p', 's', 'y'])
        rule = Rule()
        rule.add_literal(l)
        rule = ds.prune_rule(rule)
        self.assertEqual("a5 in ['c', 'f', 'm', 'p', 's', 'y']", rule.to_string())

    def test_prune_rule_3(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = DictDataset(0, df)
        l = Literal('a5', 'in', ['c', 'f', 'm', 'p', 's', 'y'])
        l2 = Literal('a3', 'in', 'e')
        rule = Rule()
        rule.add_literal(l)
        rule.add_literal(l2)
        rule = ds.prune_rule(rule)
        self.assertEqual("a5 in ['c', 'f', 'm', 'p', 's', 'y']", rule.to_string())

    # TEST GROW RULE
    def test_grow_rule_1(self):
        df = pd.read_csv('test_files/testfile_4.csv',
                         encoding='utf-8', delimiter=';')
        ds = DictDataset(0, df)
        rule = ds.grow_rule()
        self.assertEqual("Age < 8 and Sex in ['Female']", rule.to_string())

    def test_grow_rule_2(self):
        df = pd.read_csv('test_files/mushroom.csv', encoding='utf-8', delimiter=';')
        ds = DictDataset(0, df)
        rule = ds.grow_rule()
        rule = ds.make_rule(rule)
        self.assertEqual("a5 in ['c', 'f', 'm', 'p', 's', 'y']", rule.to_string())
