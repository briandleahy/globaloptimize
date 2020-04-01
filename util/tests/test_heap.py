import random
import unittest

from globaloptimization.util.heap import Heap, EmptyHeapError, heapsort


class TestHeap(unittest.TestCase):
    def test_initializes_to_zero_num_in_heap_when_empty(self):
        heap = Heap()
        self.assertEqual(heap.num_in_heap, 0)

    def test_initializes_num_in_heap_to_1_when_not_empty(self):
        heap = Heap(2)
        self.assertEqual(heap.num_in_heap, 1)

    def test_initializes_child_heaps_to_none(self):
        heap = Heap(None)
        self.assertIs(heap.left_child, None)
        self.assertIs(heap.right_child, None)

    def test_stores_value(self):
        value = 1755
        heap = Heap(value)
        self.assertIs(heap.value, value)

    def test__bubble_down_creates_new_heap(self):
        initial = 1755
        heap = Heap(initial)

        values_to_add = [1766, 1777]
        for value_to_add in values_to_add:
            heap._bubble_down(value_to_add)

        children = [heap.left_child, heap.right_child]
        for child, value in zip(children, values_to_add):
            self.assertIsInstance(child, Heap)
            self.assertIs(child.value, value)

    def test_add_to_heap_keeps_ordered_when_added_low_to_high(self):
        low = 0
        high = 1
        heap = Heap(low)
        heap.add_to_heap(high)
        self.assertEqual(heap.value, low)

    def test_add_to_heap_keeps_ordered_when_added_high_to_low(self):
        low = 0
        high = 1
        heap = Heap(high)
        heap.add_to_heap(low)
        self.assertEqual(heap.value, low)

    def test_add_to_heap_keeps_count(self):
        heap = Heap(0)
        number_total = 10
        for i in range(1, number_total):
            heap.add_to_heap(number_total)

        self.assertEqual(heap.num_in_heap, number_total)

    def test_add_to_heap_keeps_balanced(self):
        heap = Heap(0)
        number_total = 25  # needs to be odd
        for i in range(1, number_total):
            heap.add_to_heap(number_total)

        number_per_child = (number_total - 1) / 2
        # 1 in main heap, (n - 1) / 2 in sub-heaps, for n total
        self.assertEqual(heap.left_child.num_in_heap, number_per_child)
        self.assertEqual(heap.right_child.num_in_heap, number_per_child)

    def test_create_from_iterable_returns_correct_length(self):
        values = [i for i in range(18)]
        heap = Heap.create_from_iterable(values)
        self.assertEqual(heap.num_in_heap, len(values))

    def test_pop_min_returns_min_value(self):
        values = [2, 5, 9, 2, 5, 3, 0, 1, 2]  # random integers on [0, 10]
        heap = Heap.create_from_iterable(values)
        out = heap.pop_min()
        self.assertEqual(out, min(values))

    def test_pop_min_decrements_num_in_heap(self):
        num_total = 10
        heap = Heap.create_from_iterable(range(num_total))
        _ = heap.pop_min()
        self.assertEqual(heap.num_in_heap, num_total - 1)

    def test_len(self):
        num_total = 10
        heap = Heap.create_from_iterable(range(num_total))
        self.assertEqual(len(heap), heap.num_in_heap)

    def test_pop_min_bubbles_up_next(self):
        values = sorted([1, 2, 3])
        heap = Heap.create_from_iterable(values)
        first = heap.pop_min()
        second = heap.pop_min()
        third = heap.pop_min()
        popped = [first, second, third]
        self.assertEqual(values, popped)

    def test_pop_min_on_empty_heap_raises_error(self):
        heap = Heap(1)
        _ = heap.pop_min()  # heap is now empty
        self.assertRaises(EmptyHeapError, heap.pop_min)

    def test_adding_after_pop_min_on_last_element(self):
        heap = Heap(1)
        _ = heap.pop_min()  # heap is now empty
        assert len(heap) == 0

        heap.add_to_heap(2)
        self.assertEqual(len(heap), 1)


class TestHeapsort(unittest.TestCase):
    def test_heapsort(self):
        numbers = [i for i in range(1918)]
        random.seed(1919)
        random.shuffle(numbers)

        heap_sorted = heapsort(numbers)
        reg_sorted = sorted(numbers)
        self.assertEqual(heap_sorted, reg_sorted)

if __name__ == '__main__':
    unittest.main()
