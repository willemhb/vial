"""
This module provides a set of core datastructures useful for implementing ASGI frameworks
(similar to Werkzeug). It also provides general helper functions for working with urls, 
requests, etc.
"""
import re
from functools import wraps, cached_property
from itertools import chain, starmap
from dataclasses import dataclass, is_dataclass
from enum import Enum
from collections.abc import Mapping, MutableMapping, Set, MutableSet, Sequence, MutableSequence
from typing import Union, Optional, Any, Iterable
from http import HTTPStatus, cookies
from urllib import parse
from enum import Enum


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


class SparseArray(MutableSequence):
    """
    Represents an array of arbitrary dimension that's mostly empty.

    Represented by a dict whose keys are tuples, where the 0-th item
    in the tuple represents the 0-th index, the 1st item represents the
    1st index, and so on.
    """
    def __init__(self, dim):
        self._dim = dim  # represents the length of the key; a key can have fewer than than this,
                         # but by defualt they're filled in with none
        self._array = dict()

    # general sequence functions
    def __len__(self):
        return len(self._array.values())

    def __contains__(self, item):
        return item in self._array.values()

    # iteration functions
    def __iter__(self):
        def inner(d):
            for k in sorted(d):
                yield d[k]

        return inner(self._array)

    def __getitem__(self, key):
        try:
            return self._array[sparse_index(key)]

        except KeyError:
            raise IndexError

    def __setitem__(self, key, value):
        self._array[sparse_index(key)] = value
        return

    def __delitem__(self, key):
        try:
            del self._array[sparse_index(key)]

        except KeyError:
            raise IndexError

        
class AsgiTypeKeys(str, Enum):
    """
    Subclasses of this Enum associate ASGI events of a particular type with a specific
    Typed dict callable.
    """
    def __new__(cls, value, evt_dct_cls):
        """
        The actual value is the first argument, the .
        """
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._evt_cls = evt_dct_cls
        return obj

"""
URL routing.
"""
CAPTURE_PAT = re.compile(r"^<([a-zA-Z:]+)>$")
BUILTIN_FILTERS = {
    "static": r"^/([^/]+)",
    "any": r"^/(?P<%s>[^/]+)",
    "int": r"\d+",
    "float": r"(?:\d*\.\d+)|(?:\d+\.\*)",
    "path": r"(?:\w+/)*(?:\w+\.\w{1,6}){1}",
    }


def eval_pattern(parsed, filters):
    name, re_name, re_pat = parsed
    if re_pat is None:
        try:
            re_pat = filters[re_name]

        except KeyError:
            raise ValueError(f"Unknown filter name {parsed[1]}")

        else:
            return [name, re_name, re_pat]

    else:
        return parsed


def eval_patterns(parsed_path, **filters):
    """
    Return a list of Rules (either static or dynamic).
    """
    filters = filters | BUILTIN_FILTERS

    return [eval_pattern(path, filters) for path in parsed_path]


def parse_url_patterns(path):
    """
    Accept a URL pattern and parse it into its static and pattern-based elements.
    """
    components = path.strip("/").split("/")
    parsed = []

    for c in components:
        if (m := CAPTURE_PAT.match(c)):
            split = m[1].split(":")

            if len(split) == 1:
                split.extend(["any", None])

            elif len(split) == 2:
                split.append(None)

            parsed.append(split)

        else:
            parsed.append([c, "static", None])

    return parsed



def analyze_path(path, static_rules, patterns, dim):
    """
    Determine the key in a sparse array of routes for the given path.

    Each index in the sparse array should contain either a string (
    corresponds to a static rule), a number (each number corresponds
    to one of the patterns that could match), or None (the path has
    fewer components than the longest path represented in the table).
    """
    components = eval_patterns(parse_url_patterns(path))
    index = [None] * dim

    for i, component in enumerate(components):
        if component[1] == "static":
            static_rules.add(component[0])
            index[i] = component[0]

        elif component[-1] in patterns:
            index[i] = patterns.index(component[-1])

        else:
            patterns.append(component[-1])
            index[i] = len(patterns) - 1

    return tuple(index)


class NoMatch(BaseException):
    """
    Returned when a RuleTable can't match a supplied key.
    """
    pass


class RuleTable:
    """
    Taking another crack at route matching.
    """ 
    STATIC_KEY = r"^/([^/]+)"

    def __init__(self, endpoint=NoMatch, /,):
        self.patterns = self._init_patterns()
        self.endpoint = NoMatch

    @cached_property
    def static(self):
        return self.patterns[self.STATIC_KEY][-1]

    def __getitem__(self, string):
        """
        This fairly complex method searches patterns in key insertion order
        (this guarantees static will be looked up first, etc.).
        """
        if string in {"/", ""}:
            return {}, self.endpoint

        # Treat the static rule (guaranteed to be first) differently from others
        items = iter(self.patterns.values())
        p, r = next(items)

        if (m := p.match(string)):
            try:
                match = m[1]
                rest = string[m.span()[1]:]
                return r[match][rest]

            except KeyError:
                pass

        for p, r in filter(None, items):
            if (m := p.match(string)) is not None:
                try:
                    c = m.groupdict()
                    rest = string[m.span()[1]:]
                    captured, endpoint = r[rest]
                    return c | captured, endpoint

                except KeyError:
                    continue

        else:
            return NoMatch

    def __setitem__(self, path, endpoint):
        """
        Really an entry point for some sort of insert method.
        """
        rules = eval_patterns(parse_url_patterns(path))
        try:
            self._insert(endpoint, *rules)

        except ValueError:
            raise ValueError(f"Ambiguous matching for {path}: matched more than once.")

    def _insert(self, endpoint, *rules):
        if rules:
            rule, *rest = rules
            name, re_name, re_pat = rule

            if re_name == "static":
                next_table = self.static.setdefault(name, RuleTable())
                next_table._insert(endpoint, *rest)

            elif (t := self.patterns.get(re_pat)):
                t[-1]._insert(endpoint, *rest)

            else:
                new = [re.compile(r"^/(?P<%s>%s)" % (name, re_pat)), RuleTable()]
                self.patterns[re_pat] = new
                new[-1]._insert(endpoint, *rest)

        elif self.endpoint is not NoMatch:
            raise ValueError

        else:
            self.endpoint = endpoint
            return

    def _init_patterns(self):
        patterns = {v: None for v in BUILTIN_FILTERS.values()}
        patterns[self.STATIC_KEY] = [re.compile(self.STATIC_KEY), {}]

        return patterns
