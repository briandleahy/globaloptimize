import unittest

import numpy as np

from globaloptimization.geometry.simplex import Simplex, FunctionPoint


class TestFunctionPoint(unittest.TestCase):
    def test_stores_point_and_value(self):
        point = np.ones(4)
        value = 3.5
        function_point = FunctionPoint(point, value)
        self.assertEqual(function_point.value, value)
        self.assertTrue(np.all(function_point.point == point))


class TestSimplex(unittest.TestCase):
    def test_raises_error_if_points_not_all_same_dimension(self):
        n_dim = 3
        n_points = n_dim + 1
        points = np.random.standard_normal((n_points, n_dim))
        values = np.random.standard_normal(n_points + 1)
        correct_function_points = make_function_points(points[:-1], values[:-1])
        incorrect_function_point = FunctionPoint(points[-1, :-1], values[-1])
        function_points = correct_function_points + [incorrect_function_point]
        self.assertRaises(ValueError, Simplex, function_points)

    def test_raises_error_if_values_not_1d(self):
        n_dim = 3
        n_points = n_dim + 1
        points = np.random.standard_normal((n_points, n_dim))
        values = points.copy()
        function_points = make_function_points(points, values)
        self.assertRaises(ValueError, Simplex, function_points)

    def test_raises_error_if_points_incorrect_shape(self):
        n_dim = 3
        points = np.random.standard_normal((n_dim, n_dim))
        values = np.random.standard_normal(points.shape[:1])
        function_points = make_function_points(points, values)
        self.assertRaises(ValueError, Simplex, function_points)

    def test_dimension_is_correct(self):
        n_dim = 3
        points = np.random.standard_normal((n_dim + 1, n_dim))
        values = np.random.standard_normal(points.shape[:1])
        function_points = make_function_points(points, values)
        simplex = Simplex(function_points)
        self.assertEqual(simplex.dimension, n_dim)

    def test_simplex_branch_on_interior_point_returns_correct_number(self):
        np.random.seed(1104)
        dimension = 3
        simplex = make_simplex(dimension)
        new_point = np.random.standard_normal((dimension,))
        new_value = np.random.standard_normal((1,))
        new_function_point = FunctionPoint(new_point, new_value)
        branched_simplices = simplex.branch_on_interior_point(
            new_function_point)
        self.assertEqual(len(branched_simplices), dimension + 1)
        for branched_simplex in branched_simplices:
            self.assertIsInstance(branched_simplex, Simplex)

    def test_simplex_branch_on_interior_point_keeps_d_old_simplices(self):
        np.random.seed(1104)
        dimension = 3
        simplex = make_simplex(dimension)
        new_point = np.random.standard_normal((dimension,))
        new_value = np.random.standard_normal((1,))
        new_function_point = FunctionPoint(new_point, new_value)
        branched_simplices = simplex.branch_on_interior_point(
            new_function_point)

        old_function_points = simplex.function_points
        for branched_simplex in branched_simplices:
            count = 0
            these_function_points = branched_simplex.function_points
            for op in old_function_points:
                if any([op is tfp for tfp in these_function_points]):
                    count += 1
            self.assertEqual(count, dimension)

    def test_max_function_point(self):
        n_dim = 3
        points = np.random.standard_normal((n_dim + 1, n_dim))
        values = np.random.standard_normal(points.shape[:1])
        function_points = make_function_points(points, values)
        simplex = Simplex(function_points)

        max_function_point = simplex.max_function_point
        self.assertEqual(max_function_point.value, values.max())

    def test_min_function_point(self):
        n_dim = 3
        points = np.random.standard_normal((n_dim + 1, n_dim))
        values = np.random.standard_normal(points.shape[:1])
        function_points = make_function_points(points, values)
        simplex = Simplex(function_points)

        min_function_point = simplex.min_function_point
        self.assertEqual(min_function_point.value, values.min())


def make_function_points(points, values):
    return [FunctionPoint(p, v) for p, v in zip(points, values)]


def make_simplex(dimension=3):
    points = np.random.standard_normal((dimension + 1, dimension))
    values = np.random.standard_normal((dimension + 1,))
    function_points = make_function_points(points, values)
    return Simplex(function_points)


if __name__ == '__main__':
    unittest.main()
