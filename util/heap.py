from collections import deque


class Heap(object):
    """
    A data structure which efficiently keeps the min value at the top.

    Both adding an object to the heap and popping the minimum object
    from the heap take O(log(N)) operations.

    Methods
    -------
    create_from_iterable: iterable -> Heap
    add_to_heap
        Add a`
    pop_min:
        Remove and return the minimum element from the heap.

    Raises
    ------
    EmptyHeapError
        Raised when pop_min() is called on an empty heap.

    See Also
    --------
    heapsort
        Uses a Heap structure to sort items in O(N log(N)) time.
    """

    def __init__(self, value=None):
        """
        Parameters
        ----------
        value : comparison-sortable object or None, optional
            The first value to add to the heap. If None, the heap is
            initialized empty. If not None, `value` must be comparable
            to the other objects in the heap with >, <, >=, <=, and ==.
        """
        self.value = value
        self.num_in_heap = 0 if value is None else 1
        self.left_child = None
        self.right_child = None

    @classmethod
    def create_from_iterable(cls, iterable):
        as_iterable = iter(iterable)
        first_value = next(as_iterable)
        heap = cls(first_value)
        for i in as_iterable:
            heap.add_to_heap(i)
        return heap

    def add_to_heap(self, value):
        """
        Parameters
        ----------
        value : comparison-sortable object
            `value` must be comparable to the other objects in the heap
            with >, <, >=, <=, and ==.
        """
        if self.value is None:
            self.value = value
        elif value < self.value:
            self._bubble_down(self.value)
            self.value = value
        else:
            self._bubble_down(value)
        self.num_in_heap += 1

    def pop_min(self):
        if self.num_in_heap == 0:
            raise EmptyHeapError
        return self._pop_min()

    def _pop_min(self):
        out = self.value
        self.num_in_heap -= 1
        self.value = self._bubble_up()
        return out

    def _bubble_down(self, value):  # needs a better name
        if self.left_child is None:
            self.left_child = self.__class__(value)
        elif self.right_child is None:
            self.right_child = self.__class__(value)
        elif self.left_child.num_in_heap < self.right_child.num_in_heap:
            self.left_child.add_to_heap(value)
        else:
            self.right_child.add_to_heap(value)

    def _bubble_up(self):
        # 5 options: both none, 1 none (x2), neither noen
        if self.left_child is None and self.right_child is None:
            out = None
        elif self.left_child is None:
            out = self._bubble_up_right()
        elif self.right_child is None:
            out = self._bubble_up_left()
        elif self.left_child.value < self.right_child.value:
            out = self._bubble_up_left()
        else:
            out = self._bubble_up_right()
        return out

    def _bubble_up_left(self):
        out = self.left_child.pop_min()
        if len(self.left_child) == 0:
            self.left_child = None
        return out

    def _bubble_up_right(self):
        out = self.right_child.pop_min()
        if len(self.right_child) == 0:
            self.right_child = None
        return out

    def __len__(self):
        return self.num_in_heap


class EmptyHeapError(Exception):
    pass


def heapsort(x):
    iterable = iter(x)
    heap = Heap(next(iterable))
    for i in iterable:
        heap.add_to_heap(i)

    out = deque()
    while len(heap) > 0:
        out.append(heap.pop_min())
    return list(out)

