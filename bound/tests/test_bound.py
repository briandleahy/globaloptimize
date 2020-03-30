import unittest

import numpy as np

from globaloptimization.bound import bound
from globaloptimization.geometry.simplex import Simplex
from globaloptimization.util.util import ObjectValuePair
from globaloptimization.geometry.tests.test_simplex import make_simplex


class TestBoundCalculator(unittest.TestCase):
    def test_under_bound_raises_notimplementederror(self):
        bounder = bound.SimplexBoundCalculator()
        simplex = make_simplex()
        self.assertRaises(NotImplementedError, bounder._bound, simplex)

    def test_bound_checks_if_simplex(self):
        bounder = bound.SimplexBoundCalculator()
        simplex = make_simplex()
        function_point = simplex.function_points[0]
        self.assertRaises(ValueError, bounder.bound, function_point)


class TestPointBoundCalculator(unittest.TestCase):
    def test_bound_raises_notimplementederror(self):
        bounder = bound.PointBoundCalculator()
        distance = 1.0
        self.assertRaises(NotImplementedError, bounder.bound, distance)


class TestOrdinaryPointBoundCalculatorBound(unittest.TestCase):
    def test_bounds_quadratic_at_short_distances(self):
        bounder = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=1.0,
            df1_dx1_lipshitz_constant=1.0,
            )
        dx = 1e-3
        bound1 = bounder.bound(dx)
        bound2 = bounder.bound(2 * dx)
        self.assertAlmostEqual(4 * bound1, bound2, places=13)

    def test_bounds_linear_at_long_distances(self):
        bounder = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=1.0,
            df1_dx1_lipshitz_constant=1.0,
            )
        long_distance = 100
        dx = 1e-3
        bound0 = bounder.bound(long_distance)
        bound1 = bounder.bound(long_distance + dx)
        bound2 = bounder.bound(long_distance + 2 * dx)
        self.assertAlmostEqual(bound2 - bound1, bound1 - bound0, places=13)

    def test_bounds_continuous_at_cutoff_distance(self):
        np.random.seed(1432)
        bounder = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=np.random.randn(),
            df1_dx1_lipshitz_constant=np.random.randn(),
            )
        cutoff_distance = bounder._cutoff_dist
        places = 5
        dx = 10**(-places - 1)
        bound_below = bounder.bound(cutoff_distance - dx)
        bound_above = bounder.bound(cutoff_distance + dx)
        self.assertAlmostEqual(bound_below, bound_above, places)

    def test_short_distance_bound_linear_in_df1dx1_constant(self):
        bounder_low = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=100,
            df1_dx1_lipshitz_constant=1,
            )
        bounder_hi = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=100,
            df1_dx1_lipshitz_constant=2,
            )
        dx = 1e-3
        self.assertAlmostEqual(
            2 * bounder_low.bound(dx),
            bounder_hi.bound(dx),
            places=13)

    def test_long_distance_bound_linear_in_df1dx1_constant(self):
        bounder_low = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=1,
            df1_dx1_lipshitz_constant=1,
            )
        bounder_hi = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=2,
            df1_dx1_lipshitz_constant=1,
            )
        long_distance = 100

        dx = 1e-3
        slope_low = (bounder_low.bound(long_distance + dx) -
                     bounder_low.bound(long_distance))
        slope_hi = (bounder_hi.bound(long_distance + dx) -
                    bounder_hi.bound(long_distance))
        self.assertAlmostEqual(2 * slope_low, slope_hi, places=13)

    def test_valid_bounds_when_df1_dx1_constant_is_inf(self):
        f_lipshitz_constant = 1.0
        bounder = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=f_lipshitz_constant,
            df1_dx1_lipshitz_constant=np.inf,
            )
        dx = 1
        bounds = bounder.bound(dx)
        correct = f_lipshitz_constant * dx
        self.assertAlmostEqual(bounds, correct, places=13)

    def test_valid_bounds_when_f_constant_is_inf(self):
        df1_dx1_lipshitz_constant = 1.0
        bounder = bound.OrdinaryPointBoundCalculator(
            f_lipshitz_constant=np.inf,
            df1_dx1_lipshitz_constant=df1_dx1_lipshitz_constant,
            )
        dx = 1
        bounds = bounder.bound(dx)
        correct =  0.5 * df1_dx1_lipshitz_constant * dx**2
        self.assertAlmostEqual(bounds, correct, places=13)




if __name__ == '__main__':
    pass
