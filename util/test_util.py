import copy
import random
import unittest

import numpy as np

from globaloptimization.util.util import ObjectValuePair
from globaloptimization.util.heap import heapsort
from globaloptimization.geometry.simplex import Simplex, FunctionPoint


class TestObjectValuePair(unittest.TestCase):
    def test_stores_object_value(self):
        obj = 'abcdef'
        value = 1
        object_value_pair = ObjectValuePair(obj, value)
        self.assertEqual(object_value_pair.object, obj)
        self.assertEqual(object_value_pair.value, value)

    def test_eq(self):
        ovp1 = ObjectValuePair('abc', 1)
        ovp2 = ObjectValuePair('def', ovp1.value)
        self.assertEqual(ovp1, ovp2)

    def test_lt(self):
        ovp1 = ObjectValuePair('abc', 1)
        ovp2 = ObjectValuePair('def', ovp1.value + 1)
        self.assertLess(ovp1, ovp2)

    def test_gt(self):
        ovp1 = ObjectValuePair('abc', 1)
        ovp2 = ObjectValuePair('def', ovp1.value - 1)
        self.assertGreater(ovp1, ovp2)

    def test_ge(self):
        ovp1 = ObjectValuePair('abc', 1)
        ovp2 = ObjectValuePair('def', ovp1.value - 1)
        self.assertGreaterEqual(ovp1, ovp2)

    def test_le(self):
        ovp1 = ObjectValuePair('abc', 1)
        ovp2 = ObjectValuePair('def', ovp1.value + 1)
        self.assertLessEqual(ovp1, ovp2)

    def test_repr_is_executable(self):
        # and therefore legible. We do this in a way that doesn't execute
        # arbitrary code though:
        correct_repr = "ObjectValuePair(None, 1)"
        ovp1 = eval(correct_repr)
        self.assertEqual(repr(ovp1), correct_repr)

    def test_sortable(self):
        np.random.seed(1148)
        random.seed(1150)

        simplices = [make_simplex() for _ in range(20)]
        correct_order = [
            ObjectValuePair(simplex, i)
            for i, simplex in enumerate(simplices)]
        random_order = correct_order.copy()
        random.shuffle(random_order)
        sorted_order = sorted(random_order)

        for correct, check in zip(correct_order, sorted_order):
            self.assertIs(correct, check)

    def test_heapsortable(self):
        np.random.seed(1148)
        random.seed(1150)

        simplices = [make_simplex() for _ in range(20)]
        correct_order = [
            ObjectValuePair(simplex, i)
            for i, simplex in enumerate(simplices)]
        random_order = correct_order.copy()
        random.shuffle(random_order)
        sorted_order = heapsort(random_order)

        for correct, check in zip(correct_order, sorted_order):
            self.assertIs(correct, check)


def make_simplex(dimension=3):
    points = np.random.standard_normal((dimension + 1, dimension))
    values = np.random.standard_normal((dimension + 1,))
    function_points = make_function_points(points, values)
    return Simplex(function_points)


def make_function_points(points, values):
    return [FunctionPoint(p, v) for p, v in zip(points, values)]


if __name__ == '__main__':
    unittest.main()

