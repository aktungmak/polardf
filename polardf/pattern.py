from typing import Union
import polars as pl


class Var:
    def __init__(self, name=""):
        self.__name = name or str(id(self))
        self.__unique_name = f"{name}_{id(self)}"

    def name(self):
        return self.__name

    def _unique_name(self):
        return self.__unique_name


def vars(*names):
    if len(names) == 1 and isinstance(names[0], int):
        return [Var() for _ in range(names[0])]
    return [Var(name) for name in names]


TriplePattern = tuple[Union[pl.Expr, Var, str]]
GraphPattern = list[TriplePattern]


class Any:
    x = 0

    @classmethod
    def name(cls):
        cls.x += 1
        return f"Any_{cls.x}"


class TripleMatch:
    def __init__(self, s=Any, p=Any, o=Any, g=Any):
        self.s = s
        self.p = p
        self.o = o
        self.g = g


def is_variable(term) -> bool:
    return isinstance(term, Var)


def is_literal(term) -> bool:
    return isinstance(term, (str, int, float))


def _expand_pattern(pattern: tuple) -> list[tuple]:
    """
    expand a graph pattern according to the following rules:
    s-expansion:
    ([s1, s2], p, o)          -> (s1, p, o),
                                 (s2, p, o)
    po-expansion:
    (s, [(p1, o1), (p2, o2)]) -> (s, p1, o1),
                                 (s, p2, o2)
    o-expansion:
    (s, p, [o1, o2])          -> (s, p, o1),
                                 (s, p, o2)
    p-expansion:
    (s, [p1, p2], o)          -> (s, p1, t1),
                                 (t1, p2, o)

    The rules can be combined. For using a named graph, that should be
    provided as a dictionary, e.g.
    {"graph1": [patterns...], "?graph2": [patterns...]}
    """
    shape = [type(e) is list for e in pattern]
    # basic case
    if shape == [False, False, False]:
        return [pattern]
    # po-expansion
    if shape == [False, True]:
        s, po = pattern
        return [(s, p, o) for p, o in po]
    # s-expansion
    if shape == [True, False, False]:
        ss, p, o = pattern
        return [(s, p, o) for s in ss]
    # p-expansion
    elif shape == [False, True, False]:
        s, ps, o = pattern
        temp_vars = vars(len(ps) - 1)
        ss = [s] + temp_vars
        os = temp_vars + [o]
        return zip(ss, ps, os)
    # o-expansion
    elif shape == [False, False, True]:
        s, p, os = pattern
        return [(s, p, o) for o in os]
    else:
        raise SyntaxError(f"invalid pattern: {pattern}")
