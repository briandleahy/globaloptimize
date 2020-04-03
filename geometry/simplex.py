from collections import namedtuple

import numpy as np


# TODO: Some premature optimizations that are the root of all evil:
# lazily precompute simplex.min, simplex.max
# hashable FunctionPoints


"""
FunctionPoint = namedtuple(
    "FunctionPoint",
    ('point', 'value', 'is_local_minimum'),
    defaults=(False,))  # defaults is not added until 3.7
"""
class FunctionPoint(object):
    def __init__(self, point, value, is_local_minimum=False):
        self._tuple = (point, value, is_local_minimum)

    @property
    def point(self):
        return self._tuple[0]

    @property
    def value(self):
        return self._tuple[1]

    @property
    def is_local_minimum(self):
        return self._tuple[2]

    @property
    def _data(self):
        return tuple(self.point) + (self.value,)

    def __repr__(self):
        return "FunctionPoint({}, {})".format(self.point, self.value)

    def __hash__(self):
        return hash(self._data)

    def __eq__(self, other):
        return self._data == other._data


class Simplex(object):
    """
    Stores information about a function on the vertices of a simplex
    (d-dimensional analogue of a triangle).

    Attributes
    ----------
    dimension : int
    function_points : tuple of FunctionPoints
    vertex_with_max_value : FunctionPoint
    vertex_with_max_value : FunctionPoint

    Methods
    -------
    branch_on_interior_point(FunctinoPoint) -> list of d simplices
    """
    def __init__(self, function_points):
        """
        Parameters
        ----------
        function_points : list-like of FunctionPoint objects
        """
        self.function_points = tuple(function_points)
        self.dimension = np.size(self.function_points[0].point)
        self._check_inputs()

    def branch_on_interior_point(self, new_function_point):
        simplices = []
        for exclude_index in range(len(self.function_points)):
            these_function_points = (
                self.function_points[:exclude_index] +
                self.function_points[exclude_index + 1:] +
                (new_function_point,))
            simplices.append(self.__class__(these_function_points))
        return simplices

    def _check_inputs(self):
        if not all([np.size(fp.value) == 1 for fp in self.function_points]):
            msg = "Each function values must be a scalar"
            raise ValueError(msg)
        if len(self.function_points) != self.dimension + 1:
            msg = "Evaluated points must be of shape (d+1, d)"
            raise ValueError(msg)
        all_same_dimension = all([
            np.size(fp.point) == self.dimension
            for fp in self.function_points])
        if not all_same_dimension:
            msg = "All poits must be same dimension"
            raise ValueError(msg)

    @property
    def vertex_with_max_value(self):
        index = np.argmax([fp.value for fp in self.function_points])
        return self.function_points[index]

    @property
    def vertex_with_min_value(self):
        index = np.argmin([fp.value for fp in self.function_points])
        return self.function_points[index]

