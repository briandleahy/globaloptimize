from globaloptimization.geometry.simplex import Simplex



# Code to calculate bounds given a simplex.
# There are a few ways to do this.
#       - Multiple ways to bound f(x) given a value at a point
#       - multiple ways to combine these

"""
 Bounds are:
    1.  Pick simplex vertex with the largest function value. Bound using
        that max value - h(delta), where delta = length of longest side.
    2.  Let delta = radius of circumscribing d-sphere. Then pick the bound
        as min(f on vertices) - h(delta)


For each of these, we can use h(delta) as:
    a. h = Lipshitz on function only
    b. h = Lipshitz on function gradient only
    c. h = Lipshitz on both function and its gradient
We capture all of these by just implementing case (c); case (a) and (b)
can be captured from (c) by setting a bound of np.inf as one of the
lipshitz constants

Finally, there are tigher bounds when we know that a certain point is a
local minimum, and we have Lipshitz contraints on:
    - df / dx           (i.e. 2nd derivative is bounded)
    - d^2 f / dx^2      (i.3. 3rd derivative is bounded)
    - d^3 f / dx^3      (i.e. 4th derivative is bounded)
    - d^4 f / dx^4      (i.e. 5th derivative is bounded)

I imagine that, in the end, only one of these will be useful.

"""


class SimplexBoundCalculator(object):
    def bound(self, simplex):
        if not isinstance(simplex, Simplex):
            raise ValueError("simplex must be a Simplex instance")
        return self._bound(simplex)

    def _bound(self, simplex):
        raise NotImplementedError("Implement in subclass")


class PointBoundCalculator(object):
    def bound(self, distance):
        raise NotImplementedError("Implement in subclass")


class OrdinaryPointBoundCalculator(PointBoundCalculator):
    """Calculate bounds on f given for distances from ordinary points,
    as opposed to points which are a local minimum."""

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

