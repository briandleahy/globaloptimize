import unittest

import numpy as np

from globaloptimization.geometry.simplex import Simplex


class TestSimplex(unittest.TestCase):
    def test_raises_error_if_points_and_values_different_lengths(self):
        n_dim = 3
        n_points = n_dim + 1
        points = np.random.standard_normal((n_points, n_dim))
        values = np.random.standard_normal(n_points + 1)
        self.assertRaises(ValueError, Simplex, points, values)

    def test_raises_error_if_values_not_1d(self):
        n_dim = 3
        n_points = n_dim + 1
        points = np.random.standard_normal((n_points, n_dim))
        values = points.copy()
        self.assertRaises(ValueError, Simplex, points, values)

    def test_raises_error_if_points_incorrect_shape(self):
        n_dim = 3
        points = np.random.standard_normal((n_dim, n_dim))
        values = np.random.standard_normal(points.shape[:1])
        self.assertRaises(ValueError, Simplex, points, values)

    def test_dimension_is_correct(self):
        n_dim = 3
        points = np.random.standard_normal((n_dim + 1, n_dim))
        values = np.random.standard_normal(points.shape[:1])
        simplex = Simplex(points, values)
        self.assertEqual(simplex.dimension, n_dim)

    def test_branch_simplex_returns_corect_number_of_simplices(self):
        np.random.seed(1104)
        dimension = 3
        simplex = make_simplex(dimension)
        new_point = np.random.standard_normal((dimension,))
        new_value = np.random.standard_normal((1,))
        branched_simplices = simplex.branch_on(new_point, new_value)
        self.assertEqual(len(branched_simplices), dimension + 1)
        for branched_simplex in branched_simplices:
            self.assertIsInstance(branched_simplex, Simplex)

    def test_branch_simplex_keeps_at_least_d_old_simplices(self):
        np.random.seed(1104)
        dimension = 3
        simplex = make_simplex(dimension)
        new_point = np.random.standard_normal((dimension,))
        new_value = np.random.standard_normal((1,))
        branched_simplices = simplex.branch_on(new_point, new_value)

        old_points = simplex.evaluated_points
        old_values = simplex.function_values
        for branched_simplex in branched_simplices:
            count = 0
            these_points = branched_simplex.evaluated_points
            for op in old_points:
                count += np.all([tp == op for tp in these_points], axis=1).sum()
            self.assertEqual(count, dimension)

    def test_branch_simplex_does_not_scramble_points(self):
        raise NotImplementedError("This will be obviated with namedtuples")


def make_simplex(dimension=3):
    points = np.random.standard_normal((dimension + 1, dimension))
    values = np.random.standard_normal((dimension + 1,))
    return Simplex(points, values)


if __name__ == '__main__':
    unittest.main()
