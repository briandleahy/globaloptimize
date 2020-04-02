import unittest

import numpy as np

from globaloptimize.geometry.simplex import Simplex, FunctionPoint
from globaloptimize.geometry.triangulate import (
    triangulate_function_points_into_simplices)


class TestMisc(unittest.TestCase):
    def test_triangulate_function_points_returns_simplices(self):
        np.random.seed(1400)
        npoints = 25
        ndim = 5
        points = np.random.randn(npoints, ndim)
        values = np.random.randn(npoints)
        function_points = [FunctionPoint(p, v) for p, v in zip(points, values)]

        simplices = triangulate_function_points_into_simplices(function_points)
        for simplex in simplices:
            self.assertIsInstance(simplex, Simplex)

    def test_triangulate_function_points_uses_all_points(self):
        np.random.seed(1400)
        npoints = 25
        ndim = 5
        points = np.random.randn(npoints, ndim)
        values = np.random.randn(npoints)
        function_points = [FunctionPoint(p, v) for p, v in zip(points, values)]

        simplices = triangulate_function_points_into_simplices(function_points)

        for function_point in function_points:
            point_in_simplex_i = [
                function_point in s.function_points
                for s in simplices]
            self.assertTrue(any(point_in_simplex_i))


if __name__ == '__main__':
    unittest.main()
