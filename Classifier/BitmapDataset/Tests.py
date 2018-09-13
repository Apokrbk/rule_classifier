import math
from pyroaring import BitMap

import pandas as pd
import unittest

from Classifier.BitmapDataset.BitmapDataset import BitmapDataset


class TestNotebook(unittest.TestCase):



    # TEST IS ANY POS EXAMPLE
    def test_is_any_pos_example_true(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_is_any_pos_true.csv', encoding='utf-8',
                         delimiter=',')
        ds = BitmapDataset(1,df)
        self.assertEqual(True, ds.is_any_pos_example())

    def test_is_any_pos_example_false(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_is_any_pos_false.csv',
                         encoding='utf-8', delimiter=',')
        ds = BitmapDataset(1,df)
        self.assertEqual(False, ds.is_any_pos_example())



    # TEST COUNT_P_N RULES
    def test_rule_count_p_n1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        rule = BitMap()
        rule.add(2)
        rule.add(6)
        ds = BitmapDataset(1,df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(2, p)
        self.assertEqual(2, n)

    def test_rule_count_p_n2(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')

        rule = BitMap()
        rule.add(7)
        rule.add(8)
        ds = BitmapDataset(1,df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(0, p)
        self.assertEqual(5, n)

    def test_rule_count_p_n3(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')

        rule = BitMap()
        rule.add(7)
        ds = BitmapDataset(1,df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(8, p)
        self.assertEqual(8, n)

    def test_rule_count_p_n4(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        rule = BitMap()
        ds = BitmapDataset(1,df)
        p, n = ds.count_p_n_rule(rule)
        self.assertEqual(0, p)
        self.assertEqual(0, n)

    #TEST DELETE COVERED
    def test_delete_covered_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        rule = BitMap()
        ds = BitmapDataset(1,df)
        len_before = ds.length()
        ds.delete_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before, len_after)

    def test_delete_covered_2(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        rule = BitMap()
        rule.add(7)
        rule.add(8)
        ds = BitmapDataset(1,df)
        len_before = ds.length()
        ds.delete_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before-5, len_after)

    def test_delete_covered_3(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        rule = BitMap()
        rule.add(7)
        ds = BitmapDataset(1,df)
        len_before = ds.length()
        ds.delete_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before-16, len_after)

    #TEST DELETE NOT COVERED
    def test_delete_not_covered_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        rule = BitMap()
        ds = BitmapDataset(1,df)
        ds.delete_not_covered(rule)
        len_after = ds.length()
        self.assertEqual(0, len_after)

    def test_delete_not_covered_2(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        rule = BitMap()
        rule.add(7)
        ds = BitmapDataset(1,df)
        len_before = ds.length()
        ds.delete_not_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before - 16, len_after)

    def test_delete_not_covered_3(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        rule = BitMap()
        rule.add(7)
        rule.add(8)
        ds = BitmapDataset(1,df)
        len_before = ds.length()
        ds.delete_not_covered(rule)
        len_after = ds.length()
        self.assertEqual(len_before - 27, len_after)

    #TEST SPLIT INTO GROWSET PRUNESET
    def test_split_into_growset_pruneset_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        ds = BitmapDataset(1,df)
        grow, prune = ds.split_into_growset_pruneset()
        self.assertEqual(21, grow.length())
        self.assertEqual(11, prune.length())

    #TEST LENGTH
    def test_length_dataset_1(self):
        df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_kubelki.csv',
                         encoding='utf-8', delimiter=';')
        ds = BitmapDataset(1,df)
        self.assertEqual(32, ds.length())




