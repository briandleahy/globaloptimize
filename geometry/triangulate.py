import numpy as np
from scipy.spatial import Delaunay

from globaloptimize.geometry.simplex import Simplex, FunctionPoint


def triangulate_function_on_hyperrectangle(function, bounds):
    points = _produce_points_at_corners_of_hyperrectangle(bounds)
    function_points = [FunctionPoint(point, function(point))
                       for point in points]
    return triangulate_function_points_into_simplices(function_points)


def triangulate_function_points_into_simplices(function_points):
    points = [fp.point for fp in function_points]
    triangulation = Delaunay(np.asarray(points))
    simplices = []
    for indices in triangulation.simplices:
        these_points = [function_points[i] for i in indices]
        simplices.append(Simplex(these_points))
    return simplices


def _produce_points_at_corners_of_hyperrectangle(bounds):
    """
    Parameters
    ----------
    bounds : (N, 2) list-like of bounds for each parameter
    """
    ndim = len(bounds)
    origin = np.array([b[0] for b in bounds])
    side_lengths = np.array([b[1] - b[0] for b in bounds])
    sides = np.eye(ndim, dtype='float') * side_lengths.reshape(ndim, 1)

    corners = []
    for corner_id in range(2**ndim):
        # Get a binary encoding for the corner as a string, '00001' etc
        encoding = bin(corner_id).split('b')[1].rjust(ndim, '0')
        this_corner = origin.copy().astype('float')
        for side, zero_or_one in zip(sides, encoding):
            this_corner += side * int(zero_or_one)
        corners.append(this_corner)
    return np.array(corners)

