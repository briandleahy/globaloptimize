from collections import namedtuple

import numpy as np

# TODO: raise an error in branch_on if the point is outside the simplex?
# otherwise you can get a double-covering
# TODO: new methods find_center??


FunctionPoint = namedtuple("FunctionPoint", ('point', 'value'))


class Simplex(object):
    def __init__(self, function_points):
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




