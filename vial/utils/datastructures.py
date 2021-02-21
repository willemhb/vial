import re
from typing.abc import MutableMapping
from immutables import Map
from functools import cached_property


class MultiDict(MutableMapping):
    def __init__(self):
        self._backing = dict()

    def __getitem__(self, key):
        return self._backing[key][0]

    def __setitem__(self, key, value):
        values = self._backing.setdefault(key, [])
        backing.insert(0, value)
        return


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
