"""
This module provides data abstractions not already in the Python standard library.
"""
from functools import wraps, cached_property
from dataclasses import dataclass, is_dataclass
from collections.abc import Mapping, MutableMapping, Set, MutableSet
from enum import Enum
from .util import parse_url_patterns, eval_pattern
import re


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


class RuleTypes(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class Rule:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class RuleSet:
    def __init__(self, static=True):
        self._type = RuleTypes.STATIC if static == True else RuleTypes.DYNAMIC
        self._rules = {}

    def __getitem__(self, rule):
        return self._rules[rule.key]

    def __setitem__(self, rule, value):
        self._rules[rule.key] = value


class RouteTable(Mapping):
    """
    Creates a routing table (a recursive tree-structured set of routes).
    """
    
