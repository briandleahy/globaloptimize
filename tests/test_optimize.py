import warnings
import unittest

import numpy as np

from globaloptimization.optimize import BranchBoundOptimizer
from globaloptimization.util.heap import Heap
from globaloptimization.geometry.simplex import Simplex, FunctionPoint
from globaloptimization.bound.bound import (
    MaxPointSimplexBoundCalculator,
    OrdinaryPointBoundCalculator,
    )
from globaloptimization.geometry.tests.test_simplex import make_simplex


class TestBranchBoundOptimizerBasicMethods(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings('ignore')

    def tearDown(self):
        warnings.filterwarnings('default')

    def test_init_stores_objective_function(self):
        objective_function = square_distance_from_center
        initial_simplices = [make_simplex() for _ in range(10)]
        simplex_bound_calculator = make_simplex_bound_calculator()
        optimizer = BranchBoundOptimizer(
            objective_function,
            initial_simplices,
            simplex_bound_calculator)
        self.assertIs(optimizer.objective_function, objective_function)

    def test_init_stores_simplex_bound_calculator(self):
        np.random.seed(1024)
        initial_simplices = [make_simplex() for _ in range(10)]
        simplex_bound_calculator = make_simplex_bound_calculator()
        optimizer = BranchBoundOptimizer(
            square_distance_from_center,
            initial_simplices,
            simplex_bound_calculator)

        self.assertIs(
            optimizer.simplex_bounder,
            simplex_bound_calculator)

    def test_setup_heap_includes_all_simplices(self):
        np.random.seed(1024)
        initial_simplices = [make_simplex() for _ in range(10)]
        simplex_bound_calculator = make_simplex_bound_calculator()
        optimizer = BranchBoundOptimizer(
            square_distance_from_center,
            initial_simplices,
            simplex_bound_calculator)

        heap = optimizer._setup_heap(initial_simplices)
        self.assertEqual(len(initial_simplices), heap.num_in_heap)

        heap_entries = []
        while len(heap) > 0:
            heap_entries.append(heap.pop_min())
        simplices_in_heap = [entry.object for entry in heap_entries]
        for simplex in initial_simplices:
            self.assertIn(simplex, simplices_in_heap)

    def test_init_sets_up_heap(self):
        np.random.seed(1024)
        initial_simplices = [make_simplex() for _ in range(10)]
        simplex_bound_calculator = make_simplex_bound_calculator()

        optimizer = BranchBoundOptimizer(
            square_distance_from_center,
            initial_simplices,
            simplex_bound_calculator)
        self.assertIsInstance(optimizer._heap, Heap)

    def test_init_sets_current_min_function_point(self):
        np.random.seed(1045)
        initial_simplices = [make_simplex() for _ in range(10)]
        simplex_bound_calculator = make_simplex_bound_calculator()
        optimizer = BranchBoundOptimizer(
            square_distance_from_center,
            initial_simplices,
            simplex_bound_calculator)

        function_values = [
            simplex.vertex_with_min_value.value
            for simplex in initial_simplices]
        true_min_found = min(function_values)
        stored_min_found = optimizer.current_min_function_point.value

        self.assertEqual(true_min_found, stored_min_found)

    def test_evaluate_objective_function_stores_min_found_function_value(self):
        np.random.seed(1359)
        objective_function = square_distance_from_center
        ndim = 7
        points = np.random.randn(ndim + 1, ndim)
        function_points = [
            FunctionPoint(p, objective_function(p)) for p in points]
        initial_simplices = [Simplex(function_points)]
        simplex_bound_calculator = make_simplex_bound_calculator()
        optimizer = BranchBoundOptimizer(
            objective_function,
            initial_simplices,
            simplex_bound_calculator)

        assert optimizer.current_min_function_point.value > 0
        global_min_x = np.zeros(ndim)
        global_min_fp = optimizer._evaluate_function_point(global_min_x)
        assert global_min_fp.value == 0

        self.assertEqual(optimizer.current_min_function_point.value, 0)

    def test_branch_on_candidate_returns_2_simplices(self):
        np.random.seed(1053)
        simplex = make_simplex(dimension=10)
        optimizer = make_branch_bound_optimizer(simplex.dimension)
        branched = optimizer.branch_on_candidate(simplex)

        self.assertEqual(len(branched), 2)

    def test_branch_on_candidate_puts_max_vertex_in_only_1_child(self):
        np.random.seed(1157)
        simplex = make_simplex(dimension=10)
        vertex_max = simplex.vertex_with_max_value
        optimizer = make_branch_bound_optimizer(simplex.dimension)
        branched = optimizer.branch_on_candidate(simplex)

        max_in_branched = [vertex_max in b.function_points for b in branched]
        self.assertTrue(any(max_in_branched))
        self.assertFalse(all(max_in_branched))

    def test_branch_on_candidate_keeps_other_vertices(self):
        np.random.seed(1159)
        simplex = make_simplex(dimension=10)
        optimizer = make_branch_bound_optimizer(simplex.dimension)

        branched = optimizer.branch_on_candidate(simplex)
        for b in branched:
            each_in_b = [
                v in b.function_points for v in simplex.function_points]
            count_in_b = sum(each_in_b)
            self.assertEqual(count_in_b, len(simplex.function_points) - 1)

    def test_branch_on_candidate_decreases_distance_from_max_vertex(self):
        np.random.seed(1159)
        simplex = make_simplex(dimension=10)
        optimizer = make_branch_bound_optimizer(simplex.dimension)
        branched = optimizer.branch_on_candidate(simplex)

        get_max_distance = lambda simplex: max([
            np.linalg.norm(simplex.vertex_with_max_value.point - v.point)
            for v in simplex.function_points])

        max_old = get_max_distance(simplex)

        for b in branched:
            max_new = get_max_distance(b)
            self.assertLess(max_new, max_old)

    def test_process_candidate_adds_to_heap(self):
        np.random.seed(1159)
        optimizer = make_branch_bound_optimizer()

        size_before_branching = len(optimizer._heap)
        candidate = optimizer._heap.pop_min()

        optimizer.process_candidate(candidate)
        self.assertEqual(len(optimizer._heap), size_before_branching + 1)


class TestBranchBoundOptimizerOptimize(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings('error')

    def tearDown(self):
        warnings.filterwarnings('default')

    def test_optimize_does_not_call_more_than_maxiter_fevs(self):
        np.random.seed(1359)
        maxfev = 5
        optimizer = make_realistic_optimizer_with_function_call_counter()
        optimizer.optimize(max_function_evaluations=maxfev, ftol=0)

        objective_function = optimizer.objective_function
        self.assertLessEqual(objective_function.counter, maxfev)

    def test_optimize_adds_correct_number_of_simplices_to_heap(self):
        np.random.seed(1432)
        maxfev = 5
        optimizer = make_realistic_optimizer_with_function_call_counter()
        num_initial_simplices = len(optimizer._heap)

        optimizer.optimize(max_function_evaluations=maxfev, ftol=0)

        objective_function = optimizer.objective_function
        self.assertEqual(
            len(optimizer._heap),
            num_initial_simplices + objective_function.counter)

    def test_optimize_places_last_checked_simplex_back_in_heap(self):
        np.random.seed(1626)
        maxfev = 1
        optimizer = make_realistic_optimizer_with_function_call_counter()
        num_initial_simplices = len(optimizer._heap)

        # we just make a huge ftol so the function finishes before 1 iteration
        optimizer.optimize(max_function_evaluations=maxfev, ftol=1e5)
        # which means that there should have been 0 function calls:
        objective_function = optimizer.objective_function
        assert objective_function.counter == 0

        self.assertEqual(len(optimizer._heap), num_initial_simplices)

    def test_optimize_converges_when_maxiter_large(self):
        np.random.seed(1428)
        optimizer = make_realistic_optimizer_with_function_call_counter(2)
        global_min = 0.0

        ftol = 0.01
        assert (optimizer.current_min_function_point.value > ftol)
        optimizer.optimize(ftol=ftol, max_function_evaluations=200)

        self.assertLessEqual(optimizer.current_min_function_point.value, ftol)

    def test_optimize_returns_correct_dtype_when_did_not_converge(self):
        np.random.seed(1652)
        optimizer = make_realistic_optimizer_with_function_call_counter(2)
        out = optimizer.optimize(ftol=0, max_function_evaluations=1)

        self.assertIsInstance(out, FunctionPoint)

    def test_optimize_returns_correct_dtype_when_did_converge(self):
        np.random.seed(1652)
        optimizer = make_realistic_optimizer_with_function_call_counter(2)
        maxfev = 30
        out = optimizer.optimize(ftol=0.1, max_function_evaluations=maxfev)
        assert optimizer.objective_function.counter < maxfev
        self.assertIsInstance(out, FunctionPoint)

    def test_optimize_returns_best_point(self):
        np.random.seed(1652)
        optimizer = make_realistic_optimizer_with_function_call_counter(2)
        global_min = 0.0

        ftol = 0.01
        result = optimizer.optimize(ftol=ftol, max_function_evaluations=200)

        self.assertLessEqual(result.value, global_min + ftol)



class FunctionCallCounter(object):
    def __init__(self, function):
        self.function = function
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        return self.function(*args, **kwargs)


def make_realistic_optimizer_with_function_call_counter(dimension=7):
    objective_function = FunctionCallCounter(square_distance_from_center)
    # We want the simplex to enclose the global minimum, which is at 0
    # So we make a simplex centered at the origin
    points_uncentered = np.random.randn(dimension + 1, dimension)
    center = points_uncentered.mean(axis=0).reshape(1, -1)
    points = points_uncentered - center

    function_points = [
        FunctionPoint(p, objective_function(p))
        for p in points]
    initial_simplices = [Simplex(function_points)]
    simplex_bound_calculator = MaxPointSimplexBoundCalculator(
        OrdinaryPointBoundCalculator(np.inf, 2))
    optimizer = BranchBoundOptimizer(
        objective_function,
        initial_simplices,
        simplex_bound_calculator)

    objective_function.counter *= 0
    return optimizer


def make_branch_bound_optimizer(dimension=3):
    initial_simplices = [make_simplex(dimension=dimension) for _ in range(10)]
    simplex_bound_calculator = make_simplex_bound_calculator()
    optimizer = BranchBoundOptimizer(
        square_distance_from_center,
        initial_simplices,
        simplex_bound_calculator)
    return optimizer


def make_simplex_bound_calculator(
        f_lipshitz_constant=1,
        df1_dx1_lipshitz_constant=1):
    out = MaxPointSimplexBoundCalculator(
        OrdinaryPointBoundCalculator(
            f_lipshitz_constant=f_lipshitz_constant,
            df1_dx1_lipshitz_constant=df1_dx1_lipshitz_constant))
    return out


def square_distance_from_center(p):
    return np.linalg.norm(p)**2



if __name__ == '__main__':
    unittest.main()
