"""
General helper functions for working with urls, requests, etc.
"""
import re
from http import HTTPStatus, cookies
from urllib import parse
from enum import Enum


CAPTURE_PAT = re.compile(r"^<([a-zA-Z:]+)>$")


class PatternRules(Enum):
    """
    An Enum storing the builtin URL matching patterns.
    """
    INT = r"\d+"
    FLOAT = r"(?:\d*\.\d+)|(?:\d+\.\*)"
    PATH = r"(?:\w+/)*(?:\w+\.[a-zA-Z]{1,6}){1}"
    ANY = r"[^/]+"


BUILTIN_FILTERS = {r.name.lower(): r.value for r in PatternRules}

    

def eval_pattern(captured_pat):
    if len(captured_pat) == 2:
        gname, pname = captured_pat
        pat = PatternRules[pname.upper()].value

    elif len(captured_pat) == 3:
        if captured_pat[1].lower() != "re":
            raise ValueError("Whatever.")

        else:
            gname, _, pat = captured_pat

    else:
        raise ValueError("Whatever.")

    return pat, re.compile(r"(?P<%s>%s)" % (gname, pat))


def parse_url_patterns(path, **filters):
    """
    Accept a URL pattern and parse it into its static and pattern-based elements.
    """
    filters = BUILTIN_FILTERS | filters
    components = path.strip("/").split("/")
    parsed = []

    for c in components:
        if (m := CAPTURE_PAT.match(c)):
            split = m[1].split(":")

            if len(split) < 2:
                split.append("any")

            parsed.append(split)

        else:
            parsed.append(c)

    return parsed