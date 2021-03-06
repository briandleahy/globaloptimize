import numpy as np

from globaloptimize.util.heap import Heap
from globaloptimize.util.util import ObjectValuePair
from globaloptimize.geometry.simplex import Simplex, FunctionPoint


# TODO:
# The current branch_on_candidate branches on longest edge from the max
# vertex. For different bouning methods, it might make sense to do the
# longest edge from the min vertex (e.g. for min-F0-based-bounds).

# TODO: Return a more detailed return type with whether the optimization
# succeeded.


class BranchBoundOptimizer(object):
    def __init__(self, objective_function, initial_simplices, simplex_bounder):
        self.objective_function = objective_function
        self.simplex_bounder = simplex_bounder
        self._heap = self._setup_heap(initial_simplices)
        self.current_min_function_point = self._get_min_function_point(
            initial_simplices)

    def optimize(self, max_function_evaluations=1000, ftol=1e-5):
        for _ in range(max_function_evaluations):
            candidate = self._heap.pop_min()
            if candidate.value > self.current_min_function_point.value - ftol:
                # add candidate back to heap, so we can re-start easily.
                self._heap.add_to_heap(candidate)
                return self.current_min_function_point
            else:
                self.process_candidate(candidate)
        # We hit max iterations:
        return self.current_min_function_point

    def process_candidate(self, candidate):
        simplex = candidate.object

        new_simplices = self.branch_on_candidate(simplex)
        for simplex in new_simplices:
            candidate = ObjectValuePair(
                simplex, self.simplex_bounder.bound(simplex))
            self._heap.add_to_heap(candidate)

    def branch_on_candidate(self, simplex):
        # choose 2 vertices to branch off
        vertices_old = simplex.function_points
        vertex_max = simplex.vertex_with_max_value
        index_farthest = np.argmax(
            [np.linalg.norm(vertex_max.point - v.point) for v in vertices_old])
        vertex_farthest = vertices_old[index_farthest]

        other_vertices = [
            v for v in vertices_old if v not in [vertex_farthest, vertex_max]]

        # branch off the midpoint of those 2 vertices
        midpoint = 0.5 * (vertex_max.point + vertex_farthest.point)
        vertex_midpoint = self._evaluate_function_point(midpoint)

        new_candidates = (
            Simplex(other_vertices + [vertex_midpoint, vertex_max]),
            Simplex(other_vertices + [vertex_midpoint, vertex_farthest]))
        return new_candidates

    def _evaluate_function_point(self, point):
        value = self.objective_function(point)
        function_point = FunctionPoint(point, value)
        if value < self.current_min_function_point.value:
            self.current_min_function_point = function_point
        return function_point

    def _setup_heap(self, simplices):
        heap_entries = [
            ObjectValuePair(simplex, self.simplex_bounder.bound(simplex))
            for simplex in simplices]
        heap = Heap.create_from_iterable(heap_entries)
        return heap

    def _get_min_function_point(self, initial_simplices):
        min_vertices = [s.vertex_with_min_value for s in initial_simplices]
        min_values = [v.value for v in min_vertices]
        index = np.argmin(min_values)
        return min_vertices[index]

