import itertools
from functools import reduce

from pyspark.sql import Column, DataFrame, functions as F
from pyspark.sql.types import StringType

from polardf.pattern import TriplePattern, GraphPattern, is_variable, is_literal, _expand_pattern

RDF_TYPE_IRI = "rdf:type"
COLS = ("s", "p", "o", "g")
SUB, PRD, OBJ, GRF = COLS


def triple_pattern_to_df(
        triple_pattern: TriplePattern, triples: DataFrame
) -> DataFrame:
    predicates = []
    constraints = {}
    renames = {}
    df = triples
    for term, col_name in zip(triple_pattern, COLS):
        if is_variable(term):
            renames[col_name] = term._unique_name()
        elif is_literal(term):
            renames[col_name] = f"{col_name}_{id(term)}"
            constraints[col_name] = term
        # elif isinstance(term, F.col):
        #     predicates.append(term)
        else:
            raise SyntaxError(f"not a variable, literal or expression: {term}")

    # Apply filters for constraints
    for col_name, value in constraints.items():
        df = df.filter(F.col(col_name) == F.lit(value))

    # Apply predicates if any
    for predicate in predicates:
        df = df.filter(predicate)

    # Apply renames
    for old_name, new_name in renames.items():
        df = df.withColumnRenamed(old_name, new_name)

    return df


def graph_pattern_to_df(
        graph_pattern: GraphPattern, triples: DataFrame
) -> DataFrame:
    return _multiway_natural_join(
        *(
            triple_pattern_to_df(triple_pattern, triples)
            for triple_pattern in graph_pattern
        )
    )


def select(
        triples: DataFrame,
        projection,  #: Union[F.col | Var],
        where: GraphPattern,
        optional: list[GraphPattern] = [],
) -> DataFrame:
    where = itertools.chain.from_iterable(_expand_pattern(pattern) for pattern in where)

    df = graph_pattern_to_df(where, triples)
    for optional_graph_pattern in optional:
        optional_df = graph_pattern_to_df(optional_graph_pattern, triples)
        # Find common columns for join
        common_cols = list(set(df.columns) & set(optional_df.columns))
        if common_cols:
            df = df.join(optional_df, on=common_cols, how="left")
        else:
            # Cross join if no common columns
            df = df.crossJoin(optional_df)

    if isinstance(projection, (list, tuple)):
        projection = [
            p if isinstance(p, Column) else F.col(p._unique_name()).alias(p.name())
            for p in projection
        ]
        return df.select(*projection)
    elif projection == "*" or str(projection) == "col(*)":
        return df.select("*")
    else:
        return df.select(projection)


def construct(
        df: DataFrame, output: GraphPattern, where: GraphPattern
) -> DataFrame:
    selection = select(df, "*", where)

    # Create list of DataFrames from each pattern
    pattern_dfs = []
    for pattern in output:
        # Create a DataFrame with the pattern columns
        pattern_dict = dict(zip(COLS, pattern))
        pattern_df = selection.select(
            *[F.lit(v).alias(k) if isinstance(v, (str, int, float))
              else F.col(v).alias(k) if isinstance(v, str) and v in selection.columns
            else F.col(v._unique_name()).alias(k) if hasattr(v, '_unique_name')
            else F.lit(str(v)).alias(k)
              for k, v in pattern_dict.items()]
        )
        pattern_dfs.append(pattern_df)

    # Union all DataFrames
    if pattern_dfs:
        result = pattern_dfs[0]
        for pattern_df in pattern_dfs[1:]:
            result = result.unionByName(pattern_df)
        return result
    else:
        # Return empty DataFrame with correct schema
        return selection.limit(0).select(*[F.lit(None).cast(StringType()).alias(col) for col in COLS])


def _multiway_natural_join(*dfs: DataFrame) -> DataFrame:
    return reduce(
        lambda l, r: l.join(
            r, on=list(set(r.columns) & set(l.columns)), how="inner"
        ),
        dfs,
    )
