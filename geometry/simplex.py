import numpy as np

# TODO: raise an error in branch_on if the point is outside the simplex?
# otherwise you can get a double-covering
# TODO: new methods find_center??
# TODO: Make a namedtuple FunctionPoint which takes (point, value), 
# then re-do this in terms of that. It will make testing easier


class Simplex(object):
    def __init__(self, evaluated_points, function_values):
        self.evaluated_points = np.asarray(evaluated_points)
        self.function_values = np.asarray(function_values)
        self.dimension = self.evaluated_points.shape[1]
        self._check_inputs()

    def branch_on(self, new_point, new_value):
        simplices = []
        for exclude_index in range(self.evaluated_points.shape[0]):
            these_points = np.vstack([
                self.evaluated_points[:exclude_index],
                self.evaluated_points[exclude_index + 1:],
                [new_point]])
            these_values = np.hstack([
                self.function_values[:exclude_index],
                self.function_values[exclude_index + 1:],
                new_value])
            simplices.append(self.__class__(these_points, these_values))
        return simplices

    def _check_inputs(self):
        if len(self.evaluated_points) != len(self.function_values):
            msg = "len(evaluated_points) must equal len(function_values)"
            raise ValueError(msg)
        if not all([np.size(v) == 1 for v in self.function_values]):
            msg = "Function values must be of shape (d+1,)"
            raise ValueError(msg)
        if self.evaluated_points.shape[0] != self.dimension + 1:
            msg = "Evaluated points must be of shape (d+1, d)"
            raise ValueError(msg)




