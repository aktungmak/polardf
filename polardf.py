import itertools
from functools import reduce
from typing import Union
import polars as pl

RDF_TYPE_IRI = "rdf:type"
COLS = ("s", "p", "o", "g")
SUB, PRD, OBJ, GRF = COLS


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


def triple_pattern_to_df(
        triple_pattern: TriplePattern, triples: pl.DataFrame
) -> pl.DataFrame:
    predicates = []
    constraints = {}
    renames = {}
    df = triples.lazy()
    for term, col_name in zip(triple_pattern, COLS):
        if is_variable(term):
            renames[col_name] = term._unique_name()
        elif is_literal(term):
            renames[col_name] = f"{col_name}_{id(term)}"
            constraints[col_name] = term
        # elif isinstance(term, pl.Expr):
        #     predicates.append(term)
        else:
            raise SyntaxError(f"not a variable, literal or expression: {term}")

    if predicates or constraints:
        df = df.filter(*predicates, **constraints)
    if renames:
        df = df.rename(renames)
    return df


def graph_pattern_to_df(
        graph_pattern: GraphPattern, triples: pl.DataFrame
) -> pl.DataFrame:
    return _multiway_natural_join(
        *(
            triple_pattern_to_df(triple_pattern, triples)
            for triple_pattern in graph_pattern
        )
    )


def select(
        triples: pl.DataFrame,
        projection,  #: Union[pl.Expr | Var],
        where: GraphPattern,
        optional: list[GraphPattern] = [],
) -> pl.DataFrame:
    where = itertools.chain.from_iterable(_expand_pattern(pattern) for pattern in where)

    df = graph_pattern_to_df(where, triples)
    for optional_graph_pattern in optional:
        optional_df = graph_pattern_to_df(optional_graph_pattern, triples)
        df = df.join(
            optional_df,
            on=set(df.collect_schema().names())
               & set(optional_df.collect_schema().names()),
            how="left",
        )
    if isinstance(projection, (list, tuple)):
        projection = (
            p if isinstance(p, pl.Expr) else pl.col(p._unique_name()).alias(p.name())
            for p in projection
        )
    return df.select(projection)


def construct(
        df: pl.DataFrame, output: GraphPattern, where: GraphPattern
) -> pl.DataFrame:
    selection = select(df, pl.all(), where)
    return pl.concat(selection.select(**dict(zip(COLS, pattern))) for pattern in output)


def from_df(
        df: pl.DataFrame, type_: str, keys: Union[list, str], key_pattern=None
) -> pl.DataFrame:
    if type(keys) is str:
        keys = [keys]

    if key_pattern is None:
        joined = "/".join(["{}"] * len(keys))
        key_pattern = f"{type_}/{joined}"

    with_subject = df.with_columns(pl.format(key_pattern, *keys).alias(SUB)).drop(keys)
    types = with_subject.select(
        SUB, pl.lit(RDF_TYPE_IRI).alias(PRD), pl.lit(type_).alias(OBJ)
    )
    triples = with_subject.unpivot(
        index=SUB, variable_name=PRD, value_name=OBJ
    ).drop_nulls(OBJ)
    return pl.concat([types, triples])


def is_variable(term) -> bool:
    return isinstance(term, Var)


def is_literal(term) -> bool:
    return isinstance(term, (str, int, float))


def _multiway_natural_join(*dfs: pl.DataFrame) -> pl.DataFrame:
    return reduce(
        lambda l, r: l.join(
            r, on=set(r.collect_schema().names()) & set(l.collect_schema().names())
        ),
        dfs,
    )


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
