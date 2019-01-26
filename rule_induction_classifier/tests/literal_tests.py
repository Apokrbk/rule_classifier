import unittest

from rule_induction_classifier.literal import Literal


class TestNotebook(unittest.TestCase):

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
