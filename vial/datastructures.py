"""
This module provides data abstractions not already in the Python standard library.
"""
from functools import wraps, cached_property
from dataclasses import dataclass, is_dataclass
from collections.abc import Mapping, Set, MutableSet
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


class BaseRule:
    """
    A base class for URL path rules.
    """
    def __init__(self, key, /,):
        self.key = key

    def __hash__(self):
        return hash(self.key)

    def match(self, s):
        raise NotImplementedError("Subclasses must override this method.")

    def __repr__(self):
        return f"<{type(self).__name__} key='{self.key}'>"


class StaticRule(BaseRule):
    """
    Matches based on equality with the key value.
    """
    def match(self, s):
        return s == self.key


class PatternRule(BaseRule):
    def __init__(self, pattern):
        self.key, self.regex = eval_pattern(pattern)

    

class RuleSet(BaseRule):
    def __init__(self, *args):
        self.key = "__ruleset__"
        self._rules = {arg: None for arg in args}

    
    

class RuleNode:
    """
    Represents a single node in a route Tree.

    Every node has an endpoint, which it returns in response to
    the empty string, and a dictionary of rules that it tries in order.
    """
    def __init__(self, endpoint):
        """
        Set up the data attributes of the node.
        """
        self.endpoint = endpoint # The callable associated with the route at this Node
        self.rules = {
            "static": {},
        }

    def add_rule(self, rule):
        """
        Add the rule to self.static_rules or self.pattern_rules.
        """

class RouteTable(Mapping):
    """
    Creates a .
    """
