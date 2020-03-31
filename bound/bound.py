import numpy as np

from globaloptimization.geometry.simplex import Simplex


"""
There are also tigher bounds when we know that a certain point is a
local minimum, and we have Lipshitz contraints on:
    - df / dx           (i.e. 2nd derivative is bounded)
    - d^2 f / dx^2      (i.3. 3rd derivative is bounded)
    - d^3 f / dx^3      (i.e. 4th derivative is bounded)
    - d^4 f / dx^4      (i.e. 5th derivative is bounded)
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                         Bounds for simplices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SimplexBoundCalculator(object):
    def bound(self, simplex):
        if not isinstance(simplex, Simplex):
            raise ValueError("simplex must be a Simplex instance")
        return self._bound(simplex)

    def _bound(self, simplex):
        raise NotImplementedError("Implement in subclass")


class MaxPointSimplexBoundCalculator(SimplexBoundCalculator):
    """Bound as max(f) - h(max distance from argmax(f))"""

    def __init__(self, point_bound_calculator):
        self.point_bound_calculator = point_bound_calculator

    def _bound(self, simplex):
        max_vertex = simplex.vertex_with_max_value
        point = max_vertex.point
        max_distance = max([
            np.linalg.norm(point - fp.point)
            for fp in simplex.function_points])
        max_difference = self.point_bound_calculator.bound(max_distance)
        return max_vertex.value - max_difference


# TODO other bounding options:
# 1. min(f) - h(radius of circumscribing sphere)
# 2. min(f) - h(max (dist from simplex center to all vertices))
# The first one should be the tightest bound, but it will take a little
# longer to compute when there are many variables.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                 Bounds for distances from a point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PointBoundCalculator(object):
    def bound(self, distance):
        raise NotImplementedError("Implement in subclass")


class OrdinaryPointBoundCalculator(PointBoundCalculator):
    """Calculate bounds on f given for distances from points
    which are not assumed to be local minima."""

    def __init__(self, f_lipshitz_constant, df1_dx1_lipshitz_constant):
        # TODO default to inf?
        self.f_lipshitz_constant = f_lipshitz_constant
        self.df1_dx1_lipshitz_constant = df1_dx1_lipshitz_constant

        self._cutoff_dist = (f_lipshitz_constant / df1_dx1_lipshitz_constant)
        self._offset = (
            0.5 * self.f_lipshitz_constant**2 / df1_dx1_lipshitz_constant)

    def bound(self, distance):
        if distance < self._cutoff_dist:
            bound = 0.5 * self.df1_dx1_lipshitz_constant * distance**2
        else:
            bound = self.f_lipshitz_constant * distance - self._offset
        return bound

