import numpy as np
from scipy.spatial import Delaunay

from globaloptimize.geometry.simplex import Simplex, FunctionPoint


def triangulate_function_points_into_simplices(function_points):
    points = [fp.point for fp in function_points]
    triangulation = Delaunay(np.asarray(points))
    simplices = []
    for indices in triangulation.simplices:
        these_points = [function_points[i] for i in indices]
        simplices.append(Simplex(these_points))
    return simplices

