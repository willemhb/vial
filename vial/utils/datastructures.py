import re, os, sys, json
from functools import wraps, cached_property, lru_cache
from itertools import chain, starmap, zip_longest, filterfalse, tee
from dataclasses import dataclass, is_dataclass
from enum import Enum
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Set, MutableSet, Collection
from typing import Union, Optional, Any, Iterable, Callable
from inspect import signature
from http import HTTPStatus, cookies
from urllib import parse
from enum import Enum


class NoMatch(BaseException):
    pass

class NoMatchError(Exception):
    pass

class AmbiguousRouteError(Exception):
    pass

class MethodNotAllowedError(Exception):
    pass


rule_pat  = r"\{(\w+)(?::(\w+))?(?::(.+))?\}"
any_pat   = r".+"
int_pat   = r"[+-]?\d+"
float_pat = r"[+-]?\d+.\d+"
path_pat  = r"^(.*/)?(?:$|(.+?)(?:(\.[^.]*$)|$))"


class Rule:
    NAMED_RULES = {"int": (int_pat, int),
                   "float": (float_pat, float),
                   "path": (path_pat, str)}

    def __init__(self, rule_str):
        if (m := re.match(rule_pat, rule_str)) is None:
            raise ValueError("Invalid rule string.")

        n, r, p = m.groups()
        self._name = n

        if p is None:
            if r is None:
                self._re = re.compile(any_pat)
                self._cnvt = str

            elif r in self.NAMED_RULES:
                pat, cnvt = self.NAMED_RULES[r]
                self._re = re.compile(pat)
                self._cnvt = cnvt

            else:
                raise ValueError(f"Unknown pattern name {p}.")

        elif r != "re":
            raise ValueError(f"Unknown pattern name {r}.")

        else:
            self._re   = re.compile(p)
            self._cnvt = str

    @cached_property
    def pattern(self):
        return self._re.pattern
            
    def match_param(self, p):
        if self._re.fullmatch(p):
            return self._name, self._cnvt(p)

        return None


def route_signature(route: str):
    """
    .
    """
    return tuple(i if re.match(rule_pat, p) else p for i,p in enumerate(filter(None, route.split("/"))))


class Route:
    def __init__(self, rule_str, endpoint, methods=("GET")):
        self.rule_str   = rule_str
        self.endpoints  = {m: endpoint for m in methods}
        mapping         = [*filter(None, rule_str.split("/"))]
        rules           = []

        for i, p in mapping:
            if re.match(rule_pat, p):
                rules.append((i,Rule(p)))
                mapping[i] = i

        if rules:
            self.mapping = tuple(mapping)
            self.rules = tuple(rules)

        else:
            self.mapping = rule_str
            self.rules = None

    @cached_property
    def dynamic_route(self):
        return self.rules is not None

    def match_path(self, path, method):
        endpt = self.methods.get(method, None)

        if self.dynamic_route:
            return endpt, None if path == self.mapping else None

        if isinstance(path, str):
            path = (*filter(None,path.split("/")),)

        if len(path) != len(self.mapping):
            return None

        rules = iter(self.rules)
        captured = {}

        for pc, mc in zip(path,self.mapping):
            if isinstance(mc,str):
                if pc != mc:
                    return None

                continue

            _, rule = next(rules)
            if (c := rule.match_param(pc)) is None:
                return None

            param, value = c
            captured[param] = value

        return endpt, captured

    def fast_match(self, path, method):
        if (endpt := self.endpoints.get(method,None)) is None:
            return None

        if not self.dynamic_route:
            return endpt, None

        captured = {}

        for i,r in self.rules:
            if (c := r.match_param(path[i])) is None:
                return None
                        
            param, value = c
            captured[param] = value

        return endpt, captured

    def add_endpoint(self, endpoint, method):
        if method in self.methods:
            raise AmbiguousRouteError(f"Multiple endpoints matching {self.rule_str}, {method}.")

        self.methods[method] = endpoint


class RouteMap:
    def __init__(self):
        self.fixed_matches   = {}
        self.dynamic_matches = MultiDict()
        self.columns         = []

    def match_route(self, route, method):
        if (rte := self.fixed_matches.get(route, None)) is not None:
            rslt = rte.fast_match(route, method)

            if rslt is None:
                raise NoMatchError(route)

        else:
            parts = tuple(filter(None, route.split("/")))

            # replace 
            rte = tuple(p if p in self.columns[i] else i for i,p in enumerate(filter(None, route.split("/"))))

            if (rtes := self.dynamic_matches.get(rte, None, index=slice())) is None:
                raise NoMatchError(route)

            for rte in rtes:
                if (rslt := rte.fast_match(route, method)):
                    break

            else:
                raise NoMatchError(route)

        if rslt[0] is None:
            raise MethodNotAllowedError(route)

        return rslt


    def add_route(self, route):
        pass

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
Custom metaclasses.
"""


class SingletonMeta(type):
    """
    Implements singleton behavior for inheriting classes.
    """
    def __init__(cls, name, bases, namespace):
        type.__init__(cls, name, bases, namespace)
        cls.__instance__ = None
        
    def __call__(cls, *args, **kwargs):
        """
        Check the class's __instance__ attribute before creating a new instance.
        """
        if cls.__instance__ is None:
            instance = type.__call__(cls, *args, **kwargs)
            cls.__instance__ = instance
            return instance

        else:
            return cls.__instance__


class Singleton(metaclass=SingletonMeta):
    """
    Inheritance hook for SingletonMeta.
    """


class InternedMeta(type):
    """
    Implements interned behavior for inheriting classes.
    
    Interned classes require a hashable key field that's used to determine
    whether to create a new instance or return an existing one. This key field
    is called `key` by default, but can be renamed  by setting the `__key__` class
    attribute.
    """
    def __new__(mcls, name, bases, namespace):
        """
        Resolve the value of `cls.__key__` and ensure that the class's `__init__` method's
        second argument has the same name as `cls.__key__`.
        """
        if "__key__" in namespace:
            key = namespace["__key__"]

        else:
            for base in bases:
                if hasattr(base, "__key__"):
                    key = base.__key__
                    break

            else:
                raise RuntimeError("Couldn't resolve the key field of an Interned type. Make sure the class has a __key__ field or one of its parents does.")

        if "__init__" in namespace:
            init = namespace["__init__"]

        else:
            for base in bases:
                if hasattr(base, "__init__"):
                    init = base.__init__
                    break

            else:
                raise RuntimeError("Couldn't determine an __init__ method for an Interned type. Make sure the class has an __init__ method or one of its parents does.")

        try:
            exception = None
            args = list(signature(init).parameters.keys())

            if args[1] != key:
                exception = RuntimeError(f"Malformed __init__ method for Singleton type: first non-self argument must be {key}, found {args[1]}.")

        except IndexError:
            exception = RuntimeError(f"Malformed __init__ method for Singleton type: insufficient arguments to __init__ (require at least 1 non-self argument).")

        finally:
            if exception is not None:
                raise exception

            else:
                return type.__new__(mcls, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        type.__init__(cls,  name, bases, namespace)
        cls.__instances__ = {}

    def __call__(cls, key, /, *args, **kwargs):
        if key in cls.__instances__:
            return cls.__instances__[key]

        else:
            new = type.__call__(cls, key, *args, **kwargs)
            cls.__instances__[key] = new
            return new


class Interned(metaclass=InternedMeta):
    """
    Inheritance hook for InternedMeta.
    """
    __key__ = "key"

    def __init__(self, key):
        self.key = key
    

# Example usage of Interned metaclass
class Symbol(Interned):
    """
    Analogous to list Symbols; a runtime-accessible interned symbol.
    """
    def __new__(cls, key, *args, **kwargs):
        """
        Mandate that key is a string.
        """
        if not isinstance(key, str):
            raise TypeError("Symbol keys must be instances of str.")

        return super().__new__(cls, key, *args, **kwargs)

    def __init__(self, key, value=None):
        self.key = key
        self.value = value


class MixinMeta(type):
    """
    Metaclass for cooperative mixin classes.

    Classes that inherit from this class will benefit
    from two key behaviors:

        1) All Mixin __init__ methods are called cooperatively, in mro order.

        2) A Mixin records the methods it defines at class creation time in a class attribute called
           __mixin__. When a class inherits from a mixin, all __mixin__ attributes in its bases are
           compared to ensure they are pairwise unique (__init__ is an exception). I.e., Mixins ensure
           that their parent classes don't overlap, respecting the spirit of Mixins.

           Inheriting classes may still override any attributes or methods of the classes they inherit
           from.
    """
    pass


class Mixin(metaclass=MixinMeta):
    """
    Inheritance hook for MixinMeta.
    """

        
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
CAPTURE_GROUP_PAT = r"<([\w\d_]+){1}(?::([\w\d_]+))?(?(2):([\w\d_]+))?>"
NAMED_CAPTURING_PAT = r"(?P<%s>%s)"
MATCH_INT_PAT = r"-?\d+"
MATCH_FLOAT_PAT = r"-?[\d.]+"
MATCH_TO_SLASH_PAT = r"[^/]+"
MATCH_PATH_PAT = r".+?"


FILTERS = {
    "int": MATCH_INT_PAT,
    "float": MATCH_FLOAT_PAT,
    "any": MATCH_TO_SLASH_PAT,
    "path": MATCH_PATH_PAT,
}


def process_route(route):
    """
    Remove capturing groups and replace them with placeholders.
    """
    new_route = re.sub(CAPTURE_GROUP_PAT, "{}", route)
    capture_groups = re.findall(CAPTURE_GROUP_PAT, route)
    patterns = []

    for name, re_name, re_pat in capture_groups:
        if not (re_name or re_pat):
            patterns.append(NAMED_CAPTURING_PAT % (name, MATCH_TO_SLASH_PAT))

        elif re_pat == '' and re_name:
            patterns.append(NAMED_CAPTURING_PAT % (name, FILTERS.get(re_name)))

        else:
            patterns.append(NAMED_CAPTURING_PAT % (name, re_pat))

    new_route = new_route.format(*patterns)

    return "^%s$" % new_route


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
        # endpoints are organized by method to narrow down the search
        self.routes = {
            "GET": {},
            "HEAD": {},
            "POST": {},
            "PUT": {},
            "DELETE": {},
            "CONNECT": {},
            "OPTIONS": {},
            "TRACE": {},
            "PATCH": {},
        }

    def __getitem__(self, meth_path):
        # routes 
        method, path = meth_path

        for route in self.routes[method]:
            if (match := re.match(route, path)):
                captured = match.groupdict()
                return captured, self.endpoints[method][route]

        return NoMatch

    def __setitem__(self, meth_path, endpoint):
        method, path = meth_path

        if self.routes[method].get(path):
            raise ValueError(f"Ambiguous route: {path} assigned to two routes.")

        self.routes[method][path] = endpoint

    def __repr__(self):
        outstring = "RuleTable({})"
        methods = []

        for method in self.routes:
            method_strings = ", ".join(starmap("({}, {})".format, self.routes[method].items()))
            methods.append(f"{method}: [{method_strings}]")

        return outstring.format(", ".join(methods))

    __str__ = __repr__


"""
File handling utilities (for config parsing).
"""
CONFIG_HEADING_PAT = re.compile(r"^[\[](.+)[\]]$")
CONFIG_SETTING_PAT = re.compile(r"^(\w+)\s*=\s*(\w+)$")
SIMPLE_LITERAL_PAT = re.compile(r"""(?P<int>-?\d+)|(?P<bool>(?:True)|(?:False))|((?P<float>-?[\d\.]+))""")


def load_dotenv(fname: str):
    """
    Add settings from a .env file to the system environment variables.
    """
    with open(fname, "rt") as dotenv:
        for line in dotenv:
            if (l := line.strip()):
                k, v = CONFIG_SETTING_PAT.match(l).groups()
                os.environ[k] = v

    return


class Config(dict):
    """
    A simple class for loading configurations from 

    
    """
    def __init__(self, conf_files=[], *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        for conf_file in conf_files:
            if conf_file.endswith(".env"):
                load_dotenv(conf_file)

            elif conf_file.endswith(".json"):
                self.load_js(conf_file)
                
            else:
                self.load(conf_file)

    def load(self, conf_file: str):
        """
        Load configurations from the supplied configuration file, declining
        to override keys that already exist.
        """
        with open(conf_file) as config:
            for line in config:
                if (l := line.strip()):  # skip blank lines
                    if (m := CONFIG_HEADING_PAT.match(l)):
                        heading_name = m[1]
                        self[heading_name] = {}

                    elif (s := CONFIG_SETTING_PAT.match(l)):
                        k, v = s[1], s[2]

                        if (pt := SIMPLE_LITERAL_PAT.fullmatch(v)):
                            for n, p in pt.groupdict():
                                if p:
                                    if n == "int":
                                        v = int(v)

                                    elif n == "float":
                                        v = float(v)

                                    else:
                                        v = bool(v)

                        self[heading_name][k] = v

                    else:
                        raise ValueError("Failed to parse line {l} in {conf_file}")

        return

    def load_js(self, conf_file: str):
        """
        Load configurations from a json file (different parsing strategy).
        """
        with open(conf_file) as js:
            from_disk = json.load(js)

        for k in from_disk:
            if k not in self:
                self[k] = from_disk

    def __getitem__(self, key):
        """
        Fall back to checking for key in the OS environment.
        """
        try:
            return super().__getitem__(self, key)

        except KeyError as e:
            try:
                return os.environ[key]

            except Exception:
                raise e


"""
Template parser.

This implements the same (more or less) Simple Template language introduced in Bottle.
"""
PYINDENT = 4 # number of spaces to expand tabs into

class NodeType(Enum):
    """
    Represents different template tokens.
    """
    EXPR = 0
    HTML = 1
    ELIF = 2
    ELSE = 3
    END = 4
    IF = 5
    EACH = 6

    def hasmatchingend(self):
        """
        Whether or not this node expects a matching end tag.
        """
        return self._value_ > 4


def getnodetype(token: str) -> NodeType:
    """
    Determine the node type of a string.
    """
    if token.startswith("{{"):
        return NodeType.EXPR

    elif token.startswith("{%"):
        if (kwm := re.search(r"(\w+)", token)):
            kw = kwm.groups(1)
            if kw == "if":
                return NodeType.IF

            elif kw == "elif":
                return NodeType.ELIF

            elif kw == "else":
                return NodeType.ELSE

            elif kw == "each":
                return NodeType.EACH

            else:
                raise ValueError("Malformed template: unrecognized keyword.")

        else:
            raise ValueError("Malformed template: unrecognized keyword.")

    else:
        return NodeType.HTML


class Token:
    """
    Represents a single template token.
    """
    def __init__(self, value, TokenType):
        pass

    

class TemplateNode:
    """
    ABC for template nodes.
    """
    def __init__(self, value, toktype, children):
        self.value = value
        self.toktype = toktype
        self.children = children


class Template:
    """
    .
    """


def tokenize_template(tmpl: str) -> list[tuple[str, str]]:
    """
    Break a template into an annotated list of tokens.
    """
    patterns = ["{{", "}}", "{%", "%}"]
    tok_regex = r"(%s.*?%s|%s.*?%s)" % patterns

    # remove empty strings from the list and return the result
    return list(filter(None, re.split(tok_regex, tmpl)))


def parse_tokens(tokens: list[str]) -> list:
    """
    Produce a list of lists representing an AST for the tokens in `tokens`.
    """
    if not tokens:
        return []

    elif tokens[0].startswith("{{"):
        return ["expr", tokens[0][2:-2].strip(), parse_tokens(tokens[1:])]

    elif tokens[0].startswith("{%"):
        toktype = "stmt"
        endcount = 1
        children = []

        for t in tokens[1:]:
            if t == "{% end %}":
                endcount -= 1

    else:
        return ["html", tokens[0], parse_tokens[1:]]


def compile_template(tokens: list[tuple[str, str]]) -> Callable:
    """
    Compile the given list of tokens into an executable template.
    """
    code = [(0, "def __exec_template():"), (1, "_buffer = []")]
    indent_level = 1

    for t_type, t_val in tokens:
        if t_type == "expression":
            py_expr = f"_tmp = %s; _buffer.append(str(_tmp))" % t_val

        elif t_type == "statement":
            py_expr = "exec('%s', globals(), locals())" % t_val

        elif t_type == "html":
            py_expr = "_buffer.append('%s')" % t_val

        else:
            raise ValueError(f"Unrecognized token type {tok_type}.")

        code.append((indent_level, py_expr))

    code.append((indent_level, "return ''.join(_buffer)"))

    assembled = ""

    for i, c in code:
        assembled += " " * (i * PYINDENT) + c + "\n"

    compiled = compile(assembled, "<string>", "exec")

    return lambda **ctx_vars: exec(compiled, globals(), ctx_vars)


@lru_cache
def get_template(path: str) -> Callable:
    """
    Compile the template associated with the file at `path`.

    The lru_cache decorator handles caching right now, so this
    function need only handle the logic of fetching and returning
    """
    try:
        with open(path, "rt") as template:
            source = template.read()

    except FileNotFoundError as e:
        raise e

    else:
        tokens = tokenize_template(source)
        compiled_template = compile_template(tokens)

        return compiled_template


# Main entry point to the template rendering system
def render(path: str, **ctx_vars) -> str:
    template_function = get_template(path)
    return template_function(**ctx_vars)
