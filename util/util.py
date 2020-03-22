

class ObjectValuePair(object):
    """A namedtuple-like object-value pair, with comparison operators
    implemented on the value."""

    def __init__(self, the_object, value):
        """
        Parameters
        ----------
        object : object
            The object to be sorted.
        value : sortable value
            The value to use for sorting.
        """
        self.object = the_object
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __repr__(self):
        the_repr = "{}({}, {})".format(
            self.__class__.__name__,
            repr(self.object),
            repr(self.value))
        return the_repr

