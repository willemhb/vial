"""
This module provides a set of core datastructures useful for implementing ASGI frameworks
(similar to Werkzeug). It also provides general helper functions for working with urls, 
requests, etc.
"""
import re
from functools import wraps, cached_property
from itertools import chain, starmap, zip_longest, filterfalse, tee
from dataclasses import dataclass, is_dataclass
from enum import Enum
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Set, MutableSet, Collection
from typing import Union, Optional, Any, Iterable
from http import HTTPStatus, cookies
from urllib import parse
from enum import Enum


"""
Iteration utilities.
"""
def interleave(*iters):
    """
    Yield elements from a group of iterables in index order, ie:

    interleave([1,2,3],['a','b','c','d'])
    """
    values = zip_longest(*[iter(iters)]*len(iters))
    lengths = [len(i) for i in iters]

    for l, value in enumerate(values):
        for i, item in enumerate(value):
            if item is None and l > lengths[i]:
                continue

            else:
                yield item


def partition(pred, iterable):
    """
    Break set into elements that satisfy the predicate or don't.
    """
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def distinct(iterable):
    seen = set()

    for element in iterable:
        if element not in seen:
            seen.add(element)
            yield element


"""
Abstract data types.
"""
class Model:
    """
    This simple model implementation uses dataclasses and the
    __init_subclass__ hook to add Model-like behavior for relatively
    less conceptual overhead.
    """
    def __init_subclass__(cls, init=True, repr=False, eq=False, ord=False, frz=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__validate__"):
            if hasattr(cls, "__post_init__"):
                # Wrap existing __post_init__ to call __validate__ as well.
                def _post_init_wrapper(meth):
                    @wraps(meth)
                    def inner(self):
                        self.__post_init__()
                        self.__validate__()
                        return
                    return inner

                cls.__post_init__ = _post_init_wrapper(cls.__post_init__)

            else:
                cls.__post_init__ = cls.__validate__

        dataclass(init=init, repr=repr, eq=eq, order=ord, frozen=frz)(cls)


is_model = is_dataclass


"""
Abstract data types - mapping types.
"""

def merge_mappings(*args, **kwargs):
    """
    Flatten args and kwargs into a single iterable that yields tuples of
    key, value pairs.
    """
    for arg in args:
        if isinstance(arg, dict):
            for item in arg.items():
                yield item

        else:
            for item in arg:
                yield item

    for item in kwargs.items():
        yield item


class MultiDict(MutableMapping):
    """
    Allows multiple keys to be associated to the same value. Uses slice syntax to 
    distinguish between a set of keys and a specific key. 

    If a user looks up a key in a MultiDict, the slice must have one or two
    components (it must have a start and optionally a stop).

    The behavior of the MultiDict depends on the value of the second component of the
    slice.

    By default, the MultiDict behaves as though it received a slice whose first component
    is a key and whose second component is -1 (the most recent assignment to that key).
    """
    def __init__(self, *args, **kwargs):
        self._d = dict()

        for k, v in merge_mappings(*args, **kwargs):
            if k not in self._d:
                self._d[k] = [v]

            else:
                self._d[k].append(v)

    def __unpack_key(self, key, default=-1, /,):
        """
        Used for input cleaning; takes either a key or a slice and returns a tuple of
        the key and either the components of the slice or a default value of -1.
        """
        if isinstance(key, slice):
            return key.start, key.stop

        return key, default

    def __cleanup(self, key):
        """
        Remove a key if there are no longer any keys associated with it.
        """
        if len(self._d[key]):
            return

        del self._d[key]

    # Iteration methods (dict iteration methods plus valuesets)
    def __iter__(self):
        return iter(self._d)

    def keys(self):
        return self._d.keys()

    def values(self):
        return chain.from_iterable(self._d.values())

    def valuesets(self):
        return self._d.values()

    def items(self):
        """
        Yield *all* key value pairs.
        """
        def inner(d):
            for k, items in d.items():
                for i in items:
                    yield k, i

        return inner(self._d)

    def itemsets(self):
        return self._d.items()
    
    # General collection methods 
    def __len__(self):
        return len(self._d)

    def __contains__(self, item):
        return item in self._d

    def __eq__(self, other):
        if type(other) is type(self):
            return self._d == other._d

        return False

    def __ne__(self, other):
        if type(other) is not type(self):
            return True

        return self._d != other._d

    def get(self, key, default=None, /, index=-1):
        return self._d.get(key, default)[index]

    def setdefault(self, key, default):
        return self._d.setdefault(key, [default])
    
    # Indexing methods (require unpacking the key)
    def __getitem__(self, key):
        k, i = self.__unpack_key(key)
        values = self._d[k]

        return values if i is None else values[i]

    def __setitem__(self, key, new):
        k, i = self.__unpack_key(key, False)

        if k not in self._d:
            self._d[k] = [new]

        elif i is None:
            self._d[k] = list(new)

        elif i is False:
            self._d[k].append(new)

        else:
            self._d[k][i] = new

    def __delitem__(self, key):
        k, i = self.__unpack_key(key)

        if i is None: # delete the whole history
            del self._d[k]

        else:
            values = self._d[k]
            del self._d[k]
            self.__cleanup(k)

    # In-place modification methods (usually provided in pairs - one applying to the most
    # recent key, the other applying to all of them - or with an optional default argument
    # to specify the index).

    def pop(self, key, index=-1, /, ):
        values = self._d[key]
        value = values.pop(index)
        self.__cleanup(key)

        return value

    def popitem(self, key, index=-1, /, ):
        values = self._d[key]

        if index is None:
            del self._d[key]
            return key, values

        else:
            value = values[index]
            del values[index]
            self.__cleanup(key)
            return key, value

    def clear(self):
        self._d.clear()
        return

    def update(self, other):
        if isinstance(other, (dict, Mapping)):
            other = other.items()

        for k, v in other:
            self[k] = v

    # Dictionary union operator
    def __or__(self, other):
        if not isinstance(other, dict):
            return NotImplemented

        new = MutableDict(self, other)
        return new

    def __ror__(self, other):
        return self.__or__(other)

    def __ior__(self, other):
        self.update(other)
        return self

    # printing methods
    def __repr__(self):
        renderer = "{}: {}".format
        return f"MultiDict({', '.join(starmap(renderer, self.itemsets()))})"

    __str__ = __repr__


def sparse_index(coll, dim):
    filled = [None] * dim
    filled[:len(coll)] = coll
    return tuple(filled)    


class SparseArray(Collection):
    """
    Represents an array of arbitrary dimension that's mostly empty.

    Represented by a dict whose keys are tuples, where the 0-th item
    in the tuple represents the 0-th index, the 1st item represents the
    1st index, and so on.

    This is a half-assed implementation of a sparse array; a more robust
    implementation may be attempted in the future.
    """
    def __init__(self):
        self._array = dict()

    def hasindex(self, index):
        return tuple(index) in self._array

    # general sequence functions
    def __len__(self):
        return len(self._array)

    def __contains__(self, item):
        return item in self._array.values()

    # iteration functions
    def __iter__(self):
        return iter(self._array.values())

    def items(self):
        return self._array.items()

    def __getitem__(self, key):
        try:
            return self._array[tuple(key)]

        except KeyError:
            raise IndexError

    def __setitem__(self, key, value):
        self._array[tuple(key)] = value
        return

    def __delitem__(self, key):
        try:
            del self._array[tuple(key)]

        except KeyError:
            raise IndexError

    def __repr__(self):
        return f"SparseArray({repr(self._array)})"

    __str__ = __repr__


class BaseOrderedSet(Set):
    """
    Common base for OrderedSet and FrozenOrderedSet.

    implementation details:

        - Insertion order is not considered when comparing two ordered sets.

        - Insertion order is maintained when a new ordered set is created by
          merging two existing ordered sets.
    """
    def __init__(self, *args):
        self._count = 0
        self._backing = {}

        for a in chain(*args):
            self._backing.setdefault(a, self._count + 1)
            self._count += 1

    def _merge(self, other):
        """
        Return an interleaved iterator over two ordered sets.
        """
        return distinct(interleave(iter(self._backing), iter(other._backing)))

    def index(self, key):
        return self._backing[key]

    # Basic iteration methods and general sequence methods
    def __len__(self):
        return len(self._backing)

    def __contains__(self, element):
        return element in self._backing

    def __iter__(self):
        return iter(self._backing)

    # Set combination sunder methods (helpers for dunder methods)
    def _andwise_(self, other):
        return filter(lambda x: x in self and x in other, interleave(self, other))

    def _orwise_(self, other):
        return filter(lambda x: x in self or x in other, interleave(self, other))

    def _xorwise_(self, other):  
        return filter(lambda x: not (x in self and x in other), interleave(self, other))

    def _diffwise_(self, other):
        return filter(lambda x: x in self and x not in other, interleave(self, other))

    # Set comparison methods
    def __le__(self, other):
        """Subset relation (self <= other)"""
        for key in self._backing:
            if key not in other:
                return False

        return True

    def __lt__(self, other):
        """Strict subset relation (self < other)"""
        for key in self._backing:
            if key not in other:
                return False

        return len(self) < len(other)

    def __eq__(self, other):
        for key in self._backing:
            if key not in other:
                return False

        return True

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        for key in other:
            if key not in self._backing:
                return False

        return len(self) > len(other)

    def __ge__(self, other):
        for key in other:
            if key not in self._backing:
                return False

        True

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(self)})"

    __str__ = __repr__


class FrozenOrderedSet(BaseOrderedSet):
    """
    Hashable ordered set.
    """
    def __and__(self, other):
        return FrozenOrderedSet(self._andwise_(other))

    def __or__(self, other):
        return FrozenOrderedSet(self._orwise_(other))

    def __xor__(self, other):
        return FrozenOrderedSet(self._xorwise_(other))

    def __sub__(self, other):
        return FrozenOrderedSet(self._diffwise_(other))

    def isdisjoin(self, other):
        for e in self:
            if e in other:
                return False

        return True

    def __hash__(self):
        return hash(tuple([type(self), *self]))


class OrderedSet(BaseOrderedSet):
    """
    A mutable set that preserves insertion order.
    """
    def add(self, item):
        if item not in self._backing:
            self._backing[item] = self._count
            self._count += 1

    def discard(self, item):
        if item in self._backing:
            del self._backing[item]

    def remove(self, item):
        if item not in self._backing:
            raise KeyError

        del self._backing[item]

    def clear(self):
        self._backing.clear()
        self._count = 0

    def pop(self):
        try:
            key, _ = self._backing.popitem()

        except Exception:
            raise KeyError

        else:
            return key

    def recount(self):
        """
        Reset the key counts after modification.
        """
        for i, k in enumerate(self, start=1):
            self._backing[k] = i

        return

    def __and__(self, other):
        return OrderedSet(self._andwise_(other))

    def __or__(self, other):
        return OrderedSet(self._orwise_(other))

    def __xor__(self, other):
        return OrderedSet(self._xorwise_(other))

    def __sub__(self, other):
        return OrderedSet(self._diffwise_(other))
    
    def isdisjoin(self, other):
        for e in self:
            if e in other:
                return False

        return True

    # In place set combinations
    def __ior__(self, other):
        for item in other:
            self.add(item)

    def __iand__(self, other):
        for element in other:
            if element not in self:
                self.remove(item)

    def __ixor__(self, other):
        for element in other:
            if element in self:
                self.remove(element)

            else:
                self.add(element)

    def __isub__(self, other):
        for element in other:
            if element in self:
                self.remove(element)


"""
URL routing.
"""
CAPTURE_GROUP_PAT = r"<([^>]+)>"
CAPTURING_PAT = r"[^/]+"

def expand_route(route):
    """
    Replace capture groups with their corresponding regular expressions,
    keeping a record of the capture names (these will later be used to match
    names to arguments when the route function is called).
    """
    names = []
    index = []

    new_route = re.sub(CAPTURE_GROUP_PAT, CAPTURING_PAT, route)

    for route_part in route.strip("/").split("/"):
        if route_part.startswith("<"):
            names.append(route_part[1:-1])
            index.append("__matching__")

        else:
            index.append(route_part)
            names.append(None)  # Put a dummy into the list of names for all static names

    return names, tuple(index), new_route.format(*[r"[^/]+"] * len(names))


def parse_route(route, static_names):
    """
    Return an index suitable for use in a SparseArray of routes.
    """
    index = []
    captured = []

    route_parts = route.strip("/").split("/")

    for route_part in route_parts:
        if route_part not in static_names:
            index.append("__matching__")
            captured.append(route_part)

        else:
            index.append(route_part)

    return captured, tuple(index)


class NoMatch(BaseException):
    """
    Returned when a RuleTable can't match a supplied key.
    """
    pass


class RuleTable:
    """
    Stores all of the rule information associated with a routing table. 

    Instance attributes:

    static_names - a set of str representing all of the unique static names handled
                   by the routing table.
    patterns     - an ordered set of all the patterns used by the routes in this table.
    endpoints    - a sparse array of all of the endpoints mapped by this table.
    """
    def __init__(self):
        self.static_names = set()
        self.endpoints = SparseArray()

    def __setitem__(self, path, endpoint):
        names, index, expanded_path = expand_route(path)
        for element in index:
            if element != "__matching__":
                self.static_names.add(element)

        endpoint.capture_groups = names
        endpoint.expanded_path = expanded_path

        if index in self.endpoints:
            raise TypeError(f"Ambiguous route patterns: {expanded_path} matched twice.")

        self.endpoints[index] = endpoint

    def __getitem__(self, path):
        captured_items, index = parse_route(path, self.static_names)

        try:
            return captured_items, self.endpoints[index]

        except KeyError:
            raise NoMatch("Couldn't match supplied url to an existing endpoint.")

    def __repr__(self):
        return f"RuleTable({', '.join(starmap('{}: {}'.format, self.endpoints.items()))})"
