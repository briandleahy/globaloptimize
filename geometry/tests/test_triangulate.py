import unittest

import numpy as np

from globaloptimize.geometry.simplex import Simplex, FunctionPoint
from globaloptimize.geometry import triangulate


class TestTriangulate(unittest.TestCase):
    def test_triangulate_function_points_returns_simplices(self):
        np.random.seed(1400)
        npoints = 25
        ndim = 5
        points = np.random.randn(npoints, ndim)
        values = np.random.randn(npoints)
        function_points = [FunctionPoint(p, v) for p, v in zip(points, values)]

        simplices = triangulate.triangulate_function_points_into_simplices(
            function_points)
        for simplex in simplices:
            self.assertIsInstance(simplex, Simplex)

    def test_triangulate_function_points_uses_all_points(self):
        np.random.seed(1400)
        npoints = 25
        ndim = 5
        points = np.random.randn(npoints, ndim)
        values = np.random.randn(npoints)
        function_points = [FunctionPoint(p, v) for p, v in zip(points, values)]

        simplices = triangulate.triangulate_function_points_into_simplices(
            function_points)

        for function_point in function_points:
            point_in_simplex_i = [
                function_point in s.function_points
                for s in simplices]
            self.assertTrue(any(point_in_simplex_i))


class TestHyperRectangleTriangulation(unittest.TestCase):
    def test_produce_points_at_corners_of_hyperrectangle_2d(self):
        bounds = [[1, 2], [3, 4]]

        calculated = triangulate._produce_points_at_corners_of_hyperrectangle(
            bounds)
        correct = np.array([
            [bounds[0][0], bounds[1][0]],
            [bounds[0][0], bounds[1][1]],
            [bounds[0][1], bounds[1][0]],
            [bounds[0][1], bounds[1][1]]])

        self.assertTrue(np.all(calculated == correct))

    def test_produce_points_at_corners_of_hyperrectangle_correct_number(self):
        for ndim in range(1, 9):
            bounds = [[0, 1]] * ndim
            number_of_corners = 2**ndim
            out = triangulate._produce_points_at_corners_of_hyperrectangle(
                bounds)
            self.assertEqual(len(out), number_of_corners)

    def test_returntype_of_triangulate_function_on_hyperrectangle(self):
        f = lambda x: np.cos(x).sum()
        np.random.seed(947)
        bounds = np.random.randn(4, 2)

        triangulation = triangulate.triangulate_function_on_hyperrectangle(
            f, bounds)
        for simplex in triangulation:
            self.assertIsInstance(simplex, Simplex)
            for function_point in simplex.function_points:
                self.assertIsInstance(function_point, FunctionPoint)
                self.assertEqual(function_point.value, f(function_point.point))

    def test_triangulation_of_triangulate_function_on_hyperrectangle(self):
        bounds = np.array([[0, 1], [0, 1]])
        f = lambda x: np.sum(x)
        triangulation = triangulate.triangulate_function_on_hyperrectangle(
            f, bounds)

        points = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
        correct_function_points = set([FunctionPoint(p, f(p)) for p in points])
        calculated_function_points = set(
            [fp for s in triangulation for fp in s.function_points])

        self.assertEqual(calculated_function_points, correct_function_points)


if __name__ == '__main__':
    unittest.main()
