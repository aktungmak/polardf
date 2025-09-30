from typing import Union
from typing_extensions import Self


class Var:
    def __init__(self, name=""):
        self.__name = name or str(id(self))
        self.__unique_name = f"{name}_{id(self)}"

    def name(self) -> str:
        return self.__name

    def _unique_name(self) -> str:
        return self.__unique_name


def vars(*names):
    if len(names) == 1 and isinstance(names[0], int):
        return [Var() for _ in range(names[0])]
    return [Var(name) for name in names]


class Scope:
    def __getattr__(self, item):
        if item == "__name__":
            return self.name
        return Var(item)


V = Scope()


class IRI:
    def __init__(self, value):
        self.value = value

    def __truediv__(self, other):
        if isinstance(other, (IRI, Var)):
            return Path(self, other)
        elif isinstance(other, Path):
            return Path(self, *other.elements)
        else:
            raise ValueError(f"Cannot extend Path with {type(other)}")

    def __getitem__(self, item):
        # exact path length
        if isinstance(item, int):
            elements = [self] * item
            return Path(*elements)
        # variable length
        if isinstance(item, slice):
            return StarPath(item.start, item.stop)
        else:
            raise IndexError(f"Can only index with int or slice")


    def __repr__(self):
        return f"IRI({self.value})"


class Lit:
    def __init__(self, value):
        self.value = value

class StarPath:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    def __repr__(self):
        return f"StarPath({', '.join(repr(e) for e in self.elements)})"

class Path:
    def __init__(self, *elements: Union[IRI, Var]):
        self.elements = list(elements)

    def __truediv__(self, other) -> Self:
        if isinstance(other, (IRI, Var)):
            return Path(*(self.elements + [other]))
        elif isinstance(other, Path):
            return Path(*(self.elements + other.elements))
        else:
            raise ValueError(f"Cannot extend Path with {type(other)}")

    def __repr__(self):
        return f"Path({', '.join(repr(e) for e in self.elements)})"


Term = Union[IRI, Lit, Var]
Triple = tuple[IRI, IRI, Union[IRI, Lit]]
TriplePattern = tuple[Term, Term, Term]
GraphPattern = list[TriplePattern]


def t(subject, predicate, object) -> Triple:
    """Create a triple with IRIs for subject, predicate, and object."""
    return (IRI(subject), IRI(predicate), IRI(object))


def tl(subject, predicate, object) -> Triple:
    """Create a triple with IRIs for subject and predicate, and a literal for object."""
    return (IRI(subject), IRI(predicate), Lit(object))


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
    return isinstance(term, Lit)

def is_iri(term) -> bool:
    return isinstance(term, IRI)


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
