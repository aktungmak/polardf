import itertools
from typing import Dict, List, Optional, Union

from rdflib import BNode, Literal, URIRef, XSD
from rdflib.paths import Path, SequencePath, MulPath
from rdflib.plugins.sparql import parser, algebra
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import Variable
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    Float,
    select,
    create_engine,
    true,
    Engine,
    CursorResult,
    quoted_name,
    CTE,
    Select,
    and_,
    null,
    literal,
    case,
    func,
    or_,
    not_,
    asc,
    desc,
    exists,
    column,
)


def term_to_string(term) -> Optional[str]:
    """Convert an RDF term (URIRef, Literal, or BNode) to its string representation.

    Returns None if the term is not a recognised RDF term type (e.g., a Variable).

    The format matches how terms are stored in the database:
    - URIRefs: the raw URI string
    - Literals: the lexical value (type info is stored in the 'ot' column)
    - BNodes: _:id
    """
    if isinstance(term, URIRef):
        return str(term)
    elif isinstance(term, Literal):
        return str(term)
    # TODO use isinstance here?
    elif hasattr(term, "__class__") and term.__class__.__name__ == "BNode":
        return f"_:{term}"
    return None


def term_to_object_type(term) -> Optional[str]:
    """Get the object type for the 'ot' column.

    Returns:
    - None for URIRef and BNode (IRI/blank node)
    - Datatype URI string for typed literals (xsd:string for plain literals per RDF 1.1)
    - "@lang" for language-tagged literals
    - Empty string "" for plain literals (no datatype, no language)
    """
    if isinstance(term, Literal):
        if term.datatype:
            return str(term.datatype)
        elif term.language:
            return f"@{term.language}"
        else:
            return str(XSD.string)
    # TODO handle unknown case rather than silently passing through
    return None  # URIRef, BNode, or unknown


# TODO this should be somewhere else
def create_databricks_engine(
    server_hostname: str, http_path: str, access_token: str, **engine_kwargs
) -> Engine:
    engine_uri = (
        f"databricks://token:{access_token}@{server_hostname}?http_path={http_path}"
    )
    return create_engine(engine_uri, **engine_kwargs)


# Dispatch tables for expression translation
_RELATIONAL_OPS = {
    "=": lambda l, r: l == r,
    "!=": lambda l, r: l != r,
    "<": lambda l, r: l < r,
    ">": lambda l, r: l > r,
    "<=": lambda l, r: l <= r,
    ">=": lambda l, r: l >= r,
}

_BINARY_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}

# Built-in functions that take args and map directly to SQL functions
_BUILTIN_SIMPLE = {
    "STRLEN": func.length,
    "UCASE": func.upper,
    "LCASE": func.lower,
    "ABS": func.abs,
    "ROUND": func.round,
    "CEIL": func.ceil,
    "FLOOR": func.floor,
    "MD5": func.md5,
    "COALESCE": func.coalesce,
}

# Built-in functions that take no arguments
_BUILTIN_NOARG = {
    "RAND": func.random,
    "NOW": func.current_timestamp,
}

# Hash functions (all map to lowercase SQL function names)
# TODO merge these into _BUILTIN_SIMPLE ?
_HASH_FUNCS = {"SHA1", "SHA256", "SHA384", "SHA512"}

# Date/time extraction fields
_DATETIME_FIELDS = {"YEAR", "MONTH", "DAY", "HOURS", "MINUTES", "SECONDS"}

# Aggregate pattern names -> SQL functions
_AGG_FUNCS = {
    "Aggregate_Count": func.count,
    "Aggregate_Sum": func.sum,
    "Aggregate_Avg": func.avg,
    "Aggregate_Min": func.min,
    "Aggregate_Max": func.max,
    "Aggregate_Sample": func.min,  # SAMPLE falls back to MIN
}

# RDF term helpers that require schema-specific mapping
_SCHEMA_DEPENDENT_FUNCS = {
    "LANG",
    "DATATYPE",
    "ISIRI",
    "ISBLANK",
    "ISLITERAL",
    "ISNUMERIC",
}

# XSD numeric type URIs for value equality comparison
_NUMERIC_TYPES = {
    str(XSD.integer),
    str(XSD.decimal),
    str(XSD.float),
    str(XSD.double),
    str(XSD.nonPositiveInteger),
    str(XSD.negativeInteger),
    str(XSD.long),
    str(XSD.int),
    str(XSD.short),
    str(XSD.byte),
    str(XSD.nonNegativeInteger),
    str(XSD.unsignedLong),
    str(XSD.unsignedInt),
    str(XSD.unsignedShort),
    str(XSD.unsignedByte),
    str(XSD.positiveInteger),
}


def _get_var_to_column(query) -> Dict[str, Column]:
    """Build a variable->column mapping from query."""
    if isinstance(query, CTE):
        return {name: query.c[name] for name in query.c.keys()}
    elif hasattr(query, "selected_columns"):
        return {col.key: col for col in query.selected_columns}
    return {}


def _get_column_names(query) -> set:
    """Get the set of column names from a query."""
    if isinstance(query, CTE):
        return set(query.c.keys())
    elif hasattr(query, "selected_columns"):
        return set(col.key for col in query.selected_columns)
    return set()


def _ensure_cte(query):
    """Convert query to a CTE for column access via .c attribute."""
    return query if isinstance(query, CTE) else query.cte()


def _add_object_type_condition(conditions: list, ot_column, o_type: Optional[str]):
    """Add object type filtering condition to the conditions list."""
    if o_type is None:
        conditions.append(ot_column.is_(None))
    else:
        conditions.append(ot_column == o_type)


class AlgebraTranslator:
    """Translates SPARQL algebra expressions to SQLAlchemy queries."""

    def __init__(
        self, engine: Engine, table_name: str = "triples", create_table: bool = False
    ):
        self.engine = engine
        self.metadata = MetaData()
        self._cte_counter = itertools.count()

        self.table = Table(
            quoted_name(table_name, quote=False),
            self.metadata,
            Column("s", String, nullable=False, primary_key=True),
            Column("p", String, nullable=False, primary_key=True),
            Column("o", String, nullable=False, primary_key=True),
            # ot contains the type of the o column - NULL means IRI or blank node
            Column("ot", String, nullable=True, primary_key=True),
        )

        # TODO create a symbol table alterantive like this
        # SELECT t.* FROM t INNER JOIN s s1 ON t.s = s1.i AND s1.t = "one" INNER JOIN s s2 ON t.o = s2.i AND s2.t = "three";

        if create_table:
            self.metadata.create_all(self.engine)

    # ---------- Execution & Translation API ----------

    def execute(self, sparql_query: str) -> Union[CursorResult, bool]:
        """Translate and execute a SPARQL query.

        Returns:
            CursorResult for SELECT queries
            bool for ASK queries
        """
        query_type, sql_query = self.translate(sparql_query)

        if query_type == "SelectQuery":
            with self.engine.connect() as conn:
                return conn.execute(sql_query)
        elif query_type == "AskQuery":
            with self.engine.connect() as conn:
                result = conn.execute(sql_query)
                return result.scalar()  # Returns the boolean value

    def translate(self, sparql_string: str):
        """Translate a SPARQL query string to a SQLAlchemy query,
        Returns query type and sqlalchemy query object
        """
        query_tree = parser.parseQuery(sparql_string)
        query_algebra = algebra.translateQuery(query_tree).algebra
        query_type = query_algebra.name
        if query_type == "SelectQuery":
            q = self._translate_select_query(query_algebra)
        elif query_type == "AskQuery":
            q = self._translate_ask_query(query_algebra)
        else:
            raise NotImplementedError(
                f"Algebra translation not implemented for {query_algebra}"
            )
        return query_type, q

    def _translate_select_query(self, select_query: CompValue):
        assert select_query.name == "SelectQuery"
        base_query = self._translate_pattern(select_query["p"])

        # If the base query is a CTE, we need to select from it
        if isinstance(base_query, CTE):
            return select(*base_query.c).select_from(base_query)

        # Otherwise, the base query is already a complete select, just return it
        return base_query

    def _translate_ask_query(self, ask_query: CompValue):
        """Translate ASK query to SELECT EXISTS(...) AS result."""
        assert ask_query.name == "AskQuery"
        base_query = self._translate_pattern(ask_query["p"])

        # We only care that at least one result exists, so add LIMIT 1
        if isinstance(base_query, CTE):
            subq = select(literal(1)).select_from(base_query).limit(1)
        else:
            subq = base_query.limit(1)

        # Wrap with EXISTS to get a boolean result
        return select(exists(subq).label("result"))

    def _translate_pattern(self, pattern):
        """Translate different pattern types."""
        if hasattr(pattern, "name"):
            if pattern.name == "BGP":
                return self._translate_bgp(pattern)
            elif pattern.name == "Project":
                return self._translate_project(pattern)
            elif pattern.name == "LeftJoin":
                return self._translate_left_join(pattern)
            elif pattern.name == "Distinct":
                return self._translate_distinct(pattern)
            elif pattern.name == "Extend":
                return self._translate_extend(pattern)
            elif pattern.name == "Filter":
                return self._translate_filter(pattern)
            elif pattern.name == "OrderBy":
                return self._translate_order_by(pattern)
            elif pattern.name == "Join":
                return self._translate_join(pattern)
            elif pattern.name == "Slice":
                return self._translate_slice(pattern)
            elif pattern.name == "Union":
                return self._translate_union(pattern)
            elif pattern.name == "Group":
                return self._translate_group(pattern)
            elif pattern.name == "AggregateJoin":
                return self._translate_aggregate_join(pattern)
            else:
                raise NotImplementedError(
                    f"Pattern type {pattern.name} not implemented"
                )
        else:
            raise ValueError(f"Unknown pattern type: {pattern}")

    def _translate_bgp(self, bgp: CompValue) -> Union[Select, CTE]:
        """Translate a Basic Graph Pattern to a query."""
        assert bgp.name == "BGP"
        triples = bgp["triples"]

        # Empty BGP = single solution with no bindings = SELECT 1
        if not triples:
            return select(column("1"))

        queries = []
        for triple in triples:
            queries.extend(self._expand_triple(triple))

        return self._join_queries(queries)

    def _expand_triple(self, triple: tuple) -> List[Select]:
        """Expand a triple pattern into one or more Select queries.

        Handles:
        - Simple triples: returns a single query
        - SequencePath (e.g. :a/:b/:c): returns queries for each step
        - MulPath (e.g. :a*): returns a recursive CTE query
        """
        s, p, o = triple

        if not isinstance(p, Path):
            return [self._triple_to_query(triple)]

        # SequencePath (e.g. :a/:b/:c)
        elif isinstance(p, SequencePath):
            temp_vars = [Variable(f"_path_{id(p)}_{i}") for i in range(len(p.args) - 1)]
            subjects = [s] + temp_vars
            predicates = p.args
            objects = temp_vars + [o]
            expanded_triples = list(zip(subjects, predicates, objects))
            return [self._triple_to_query(t) for t in expanded_triples]

        # MulPath (e.g. :a*)
        elif isinstance(p, MulPath):
            return [self._mulpath_to_cte(s, p, o)]
        else:
            raise NotImplementedError(f"Path type {type(p)} not implemented")

    def _triple_to_query(self, triple: tuple) -> Select:
        """Convert a single, expanded triple pattern to a query."""
        s, p, o = triple

        # Collect SELECT columns and WHERE conditions
        columns = []
        conditions = []

        # Subject
        if isinstance(s, Variable):
            columns.append(self.table.c.s.label(str(s)))
        elif isinstance(s, BNode):
            # Treat as unnamed variable with internal label
            # TODO this is wrong I think... it should be matching on bnodes?
            columns.append(self.table.c.s.label(f"_bnode_{s}"))
        elif (s_str := term_to_string(s)) is not None:
            conditions.append(self.table.c.s == s_str)
        else:
            raise NotImplementedError(f"Subject type {type(s)} not implemented")

        # Predicate
        if isinstance(p, Variable):
            if p == s:  # Same variable as subject
                conditions.append(self.table.c.p == self.table.c.s)
            else:
                columns.append(self.table.c.p.label(str(p)))
        elif isinstance(p, BNode):
            # TODO this is also potentially incorrect
            columns.append(self.table.c.p.label(f"_bnode_{p}"))
        elif (p_str := term_to_string(p)) is not None:
            conditions.append(self.table.c.p == p_str)
        else:
            raise NotImplementedError(f"Predicate type {type(p)} not implemented")

        # Object
        if isinstance(o, Variable):
            if o == s:  # Same variable as subject
                conditions.append(self.table.c.o == self.table.c.s)
                _add_object_type_condition(conditions, self.table.c.ot, None)
            elif o == p:  # Same variable as predicate
                conditions.append(self.table.c.o == self.table.c.p)
                _add_object_type_condition(conditions, self.table.c.ot, None)
            else:
                columns.append(self.table.c.o.label(str(o)))
                # Also select the ot column for type information (needed for value equality)
                columns.append(self.table.c.ot.label(f"_ot_{o}"))
        elif isinstance(o, BNode):
            # TODO this is also potentially incorrect
            columns.append(self.table.c.o.label(f"_bnode_{o}"))
        elif (o_str := term_to_string(o)) is not None:
            conditions.append(self.table.c.o == o_str)
            o_type = term_to_object_type(o)
            _add_object_type_condition(conditions, self.table.c.ot, o_type)
        else:
            raise NotImplementedError(f"Object type {type(o)} not implemented")

        # If no variables to select, we still need a valid SELECT clause
        # This happens when all triple positions are bound (an existence check)
        if not columns:
            columns = [literal(1).label("_exists_")]

        query = select(*columns).select_from(self.table)
        if conditions:
            query = query.where(*conditions)

        return query

    def _mulpath_to_cte(self, s, mulpath: MulPath, o):
        """Generate a recursive CTE for MulPath evaluation.

        Generates a transitive closure query, optionally filtered by bound endpoints.
        """
        # Get the predicate to follow
        # TODO handle the case when this is a nested path (see 1.1 test pp02)
        pred_value = term_to_string(mulpath.path)
        if pred_value is None:
            raise NotImplementedError(
                f"MulPath with path type {type(mulpath.path)} not implemented"
            )

        # Convert bound values to strings for comparison
        s_value = term_to_string(s)
        o_value = term_to_string(o)
        o_type = term_to_object_type(o) if not isinstance(o, Variable) else None

        # Base case: all direct edges with the predicate (need s, o, and ot for type filtering)
        base_query = select(self.table.c.s, self.table.c.o, self.table.c.ot).where(
            self.table.c.p == pred_value
        )

        # Create recursive CTE with unique name
        cte_name = f"path_cte_{next(self._cte_counter)}"
        path_cte = base_query.cte(name=cte_name, recursive=True)

        # Recursive case: extend paths forward
        recursive_query = (
            select(
                path_cte.c.s,  # Keep original starting node
                self.table.c.o,  # Extend to new destinations
                self.table.c.ot,  # Carry forward object type
            )
            .select_from(path_cte.join(self.table, path_cte.c.o == self.table.c.s))
            .where(self.table.c.p == pred_value)
        )

        # Union base and recursive parts
        path_cte = path_cte.union(recursive_query)

        # Build result columns with variable labels
        result_columns = []
        if isinstance(s, Variable):
            result_columns.append(path_cte.c.s.label(str(s)))
        if isinstance(o, Variable):
            result_columns.append(path_cte.c.o.label(str(o)))

        # If no variables, still need to select something for filtering
        if not result_columns:
            result_columns = [path_cte.c.s, path_cte.c.o]

        # Build final query with filters for bound endpoints
        final_query = select(*result_columns).select_from(path_cte)

        conditions = []
        if s_value is not None:
            conditions.append(path_cte.c.s == s_value)
        if o_value is not None:
            conditions.append(path_cte.c.o == o_value)
            _add_object_type_condition(conditions, path_cte.c.ot, o_type)

        if conditions:
            final_query = final_query.where(and_(*conditions))

        return final_query

    def _join_queries(self, queries: list):
        """Natural join multiple queries on common variables.

        Uses SPARQL-compatible join semantics where unbound variables (NULL)
        are treated as wildcards that match anything, not as distinct values.
        """
        if len(queries) == 1:
            return queries[0]

        ctes = [_ensure_cte(q) for q in queries]

        def sparql_join_cond(left_col, right_col, is_internal):
            """SPARQL join: NULL matches anything for variables, strict for internals."""
            if is_internal:
                return left_col == right_col
            return or_(left_col.is_(None), right_col.is_(None), left_col == right_col)

        # Join conditions for all pairs of CTEs on common columns
        conditions = [
            sparql_join_cond(left.c[col], right.c[col], col.startswith("_"))
            for left, right in itertools.combinations(ctes, 2)
            for col in set(left.c.keys()) & set(right.c.keys())
        ]

        # Collect all column names and identify which appear in multiple CTEs
        all_col_names = {name for cte in ctes for name in cte.c.keys()}
        cte_cols = {
            name: [cte.c[name] for cte in ctes if name in cte.c.keys()]
            for name in all_col_names
        }

        # Project columns: COALESCE for shared variables, direct reference otherwise
        def project_col(name, cols):
            if len(cols) > 1 and not name.startswith("_"):
                return func.coalesce(*cols).label(name)
            return cols[0]

        final_cols = [project_col(name, cols) for name, cols in cte_cols.items()]

        return select(*final_cols).select_from(*ctes).where(*conditions)

    def _translate_project(self, project: CompValue):
        """Translate a Project (SELECT) operation."""
        assert project.name == "Project"
        project_vars = project["PV"]
        pattern = project["p"]

        # Translate the inner pattern
        base_query = self._translate_pattern(pattern)
        base_cols = _get_column_names(base_query)
        project_var_names = set(str(var) for var in project_vars)

        if base_cols == project_var_names:
            # Projection doesn't change anything, return base query
            return base_query

        # Handle empty projections (e.g. when all terms are bound)
        if len(project_vars) == 0:
            # For empty projections, return base query as-is
            # This allows MulPath queries with bound endpoints to still work
            return base_query

        # Build projection columns
        # - Variables that exist in base query: select from the query's columns
        # - Variables that don't exist: add as NULL (unbound in SPARQL)
        if isinstance(base_query, CTE):
            # For CTEs, we need to select from them
            var_columns = [
                (
                    base_query.c[str(var)]
                    if str(var) in base_cols
                    else null().label(str(var))
                )
                for var in project_vars
            ]
            return select(*var_columns).select_from(base_query)
        else:
            # For Select objects, use with_only_columns to preserve ORDER BY
            var_to_col = {col.key: col for col in base_query.selected_columns}
            var_columns = [
                (
                    var_to_col[str(var)]
                    if str(var) in var_to_col
                    else null().label(str(var))
                )
                for var in project_vars
            ]
            return base_query.with_only_columns(*var_columns)

    def _translate_distinct(self, distinct: CompValue):
        """Translate a Distinct operation."""
        assert distinct.name == "Distinct"
        inner_query = self._translate_pattern(distinct["p"])

        # If it's a CTE, select from it with distinct
        if isinstance(inner_query, CTE):
            return select(*inner_query.c).select_from(inner_query).distinct()

        # Otherwise apply distinct directly
        return inner_query.distinct()

    def _translate_extend(self, extend: CompValue):
        """Translate an Extend (BIND) operation.

        The Extend node adds a new variable binding computed from an expression.
        Structure: Extend(p=inner_pattern, var=new_variable_name, expr=expression)
        """
        assert extend.name == "Extend"
        inner_query = self._translate_pattern(extend["p"])
        cte = _ensure_cte(inner_query)
        var_to_column = _get_var_to_column(cte)

        sql_expr = self._translate_expr(extend["expr"], var_to_column)
        result_columns = list(cte.c) + [sql_expr.label(str(extend["var"]))]

        return select(*result_columns).select_from(cte)

    def _translate_filter(self, filter_: CompValue):
        """Translate a Filter operation.

        Filter applies a boolean expression to filter results from the inner pattern.
        Structure: Filter(p=inner_pattern, expr=boolean_expression)
        """
        assert filter_.name == "Filter"
        inner_query = self._translate_pattern(filter_["p"])

        # Build var_to_column mapping from available columns
        if isinstance(inner_query, CTE):
            var_to_column = _get_var_to_column(inner_query)
            base_select = select(*inner_query.c).select_from(inner_query)
        else:
            var_to_column = _get_var_to_column(inner_query)
            base_select = inner_query

        # Translate the filter expression
        sql_condition = self._translate_expr(filter_["expr"], var_to_column)

        # Add WHERE clause directly (no extra subquery)
        return base_select.where(sql_condition)

    def _translate_order_by(self, order_by: CompValue):
        """Translate an OrderBy operation.

        OrderBy applies ordering to results from the inner pattern.
        Structure: OrderBy(p=inner_pattern, expr=list_of_order_conditions)
        """
        assert order_by.name == "OrderBy"
        inner_query = self._translate_pattern(order_by["p"])

        # Build var_to_column for expression translation
        if isinstance(inner_query, CTE):
            var_to_column = _get_var_to_column(inner_query)
            base_select = select(*inner_query.c).select_from(inner_query)
        else:
            var_to_column = _get_var_to_column(inner_query)
            base_select = inner_query

        # Build ORDER BY clauses
        order_clauses = []
        for cond in order_by["expr"]:
            if isinstance(cond, CompValue) and cond.name == "OrderCondition":
                sql_expr = self._translate_expr(cond.expr, var_to_column)
                if cond.order == "DESC":
                    order_clauses.append(desc(sql_expr))
                else:
                    order_clauses.append(asc(sql_expr))
            else:
                # Just a variable or expression (default ASC)
                sql_expr = self._translate_expr(cond, var_to_column)
                order_clauses.append(asc(sql_expr))

        return base_select.order_by(*order_clauses)

    def _translate_join(self, join: CompValue):
        """Translate a Join operation (natural join of two patterns).

        Join combines two graph patterns using natural join semantics -
        matching rows on common variables.
        Structure: Join(p1=left_pattern, p2=right_pattern)
        """
        assert join.name == "Join"
        left_query = self._translate_pattern(join["p1"])
        right_query = self._translate_pattern(join["p2"])

        return self._join_queries([left_query, right_query])

    def _translate_slice(self, slice_: CompValue):
        """Translate a Slice (LIMIT/OFFSET) operation.

        Slice limits the number of results and/or skips some results.
        Structure: Slice(p=inner_pattern, start=offset, length=limit)
        """
        assert slice_.name == "Slice"
        start = getattr(slice_, "start", 0)
        length = getattr(slice_, "length", None)

        inner_query = self._translate_pattern(slice_["p"])

        # If it's a CTE, we need to select from it first
        if isinstance(inner_query, CTE):
            result_query = select(*inner_query.c).select_from(inner_query)
        else:
            result_query = inner_query

        # Apply LIMIT
        if length is not None:
            result_query = result_query.limit(length)

        # Apply OFFSET
        if start > 0:
            result_query = result_query.offset(start)

        return result_query

    def _translate_left_join(self, left_join: CompValue):
        """Translate a LeftJoin (OPTIONAL) operation.

        In SPARQL, OPTIONAL { pattern FILTER(expr) } becomes:
        LeftJoin(p1, p2, expr) where expr is the filter condition.

        The expr is applied as part of the ON condition of the LEFT JOIN,
        NOT as a WHERE clause (which would filter out rows that don't match).
        """
        assert left_join.name == "LeftJoin"
        left_query = self._translate_pattern(left_join["p1"])
        right_query = self._translate_pattern(left_join["p2"])
        filter_expr = left_join.get("expr")

        left_cte = _ensure_cte(left_query)
        right_cte = _ensure_cte(right_query)

        # Build var_to_column from both sides (left takes precedence for common cols)
        var_to_column = {**right_cte.c, **left_cte.c}

        # Find common variables for join condition
        common_cols = set(left_cte.c.keys()) & set(right_cte.c.keys())

        # Build join conditions from common columns
        join_conditions = [left_cte.c[col] == right_cte.c[col] for col in common_cols]

        # Add the filter expression to the join conditions if present
        # (TrueFilter means no actual filter - just matches everything)
        if filter_expr is not None:
            if isinstance(filter_expr, CompValue) and filter_expr.name == "TrueFilter":
                pass  # No additional condition needed
            else:
                sql_filter = self._translate_expr(filter_expr, var_to_column)
                join_conditions.append(sql_filter)

        # All columns from both sides (deduplicated - common cols only from left)
        all_columns = list(left_cte.c) + [
            col for name, col in right_cte.c.items() if name not in common_cols
        ]

        # Build the ON condition
        if join_conditions:
            on_condition = and_(*join_conditions)
        else:
            on_condition = true()

        # LEFT JOIN
        return select(*all_columns).select_from(
            left_cte.outerjoin(right_cte, on_condition)
        )

    def _translate_union(self, union: CompValue):
        """Translate a Union operation (SPARQL UNION -> SQL UNION ALL)."""
        assert union.name == "Union"
        left_q, right_q = (self._translate_pattern(union[k]) for k in ("p1", "p2"))
        left_cte = _ensure_cte(left_q)
        right_cte = _ensure_cte(right_q)

        all_cols = sorted(_get_column_names(left_q) | _get_column_names(right_q))

        def padded_select(cte):
            """Select all columns, padding missing ones with NULL."""
            return select(
                *[
                    cte.c[c].label(c) if c in cte.c.keys() else null().label(c)
                    for c in all_cols
                ]
            ).select_from(cte)

        return padded_select(left_cte).union_all(padded_select(right_cte)).cte()

    def _translate_group(self, group: CompValue):
        """Translate a Group operation."""
        assert group.name == "Group"
        return self._translate_pattern(group["p"])

    def _translate_aggregate_join(self, agg_join: CompValue):
        """Translate an AggregateJoin (GROUP BY with aggregates).

        AggregateJoin outputs only aggregate result variables
        Grouping columns are recreated by outer Extend operations.
        """
        assert agg_join.name == "AggregateJoin"
        inner_query = self._translate_pattern(agg_join["p"])
        cte = _ensure_cte(inner_query)
        var_to_column = _get_var_to_column(cte)

        # Grouping variables determine GROUP BY clause
        group_vars = getattr(agg_join["p"], "expr", None) or []
        group_cols = [cte.c[str(v)] for v in group_vars if str(v) in var_to_column]

        # Output only aggregate result columns
        agg_cols = [
            self._translate_expr(agg, var_to_column).label(str(agg.res))
            for agg in agg_join["A"]
        ]

        result_query = select(*agg_cols).select_from(cte)
        return result_query.group_by(*group_cols) if group_cols else result_query

    # ---------- Expression Translation ----------

    def _translate_expr(self, expr, var_to_column):
        """Translate an rdflib SPARQL algebra expression (CompValue / term)
        into a SQLAlchemy expression, maintaining SPARQL semantics.

        Args:
            expr: The expression to translate (Variable, Literal, or CompValue)
            var_to_column: A dict mapping variable names to SQLAlchemy columns
        """
        # Variable - look up in var_to_column
        if isinstance(expr, Variable):
            var_name = str(expr)
            if var_name in var_to_column:
                return var_to_column[var_name]
            # Unbound variable - return NULL to use SQL's NULL comparison semantics
            # In SPARQL, comparisons with unbound variables produce errors, which are
            # treated as FALSE in filter contexts. SQL's NULL comparison semantics
            # achieve the same result: NULL = x evaluates to UNKNOWN, filtered out.
            return null()

        # Literal - handle booleans and numerics specially, else stringify
        elif isinstance(expr, Literal):
            if expr.datatype == XSD.boolean:
                return true() if expr.toPython() else not_(true())
            if expr.datatype and str(expr.datatype) in _NUMERIC_TYPES:
                return literal(expr.toPython())
            return literal(str(expr))

        elif isinstance(expr, URIRef):
            return literal(str(expr))

        elif not isinstance(expr, CompValue):
            # raw constant or already an expression
            # TODO check how we can end up here - it seems like we could miss errors here
            return literal(expr)

        name = expr.name

        # Dispatch to specific handlers
        if name == "RelationalExpression":
            return self._translate_relational(expr, var_to_column)
        elif name == "ConditionalAndExpression":
            return self._translate_conditional_and(expr, var_to_column)
        elif name == "ConditionalOrExpression":
            return self._translate_conditional_or(expr, var_to_column)
        elif name == "UnaryNot":
            return not_(self._translate_expr(expr.expr, var_to_column))
        elif name in ("AdditiveExpression", "MultiplicativeExpression"):
            return self._translate_binary_chain(expr, var_to_column)
        elif name == "InExpression":
            return self._translate_in_expression(expr, var_to_column)
        elif name.startswith("Builtin_"):
            return self._translate_builtin(expr, var_to_column)
        elif name.startswith("Aggregate_"):
            return self._translate_aggregate_expr(expr, var_to_column)
        else:
            raise NotImplementedError(f"Expression kind {name!r} not implemented")

    def _translate_relational(self, expr, var_to_column):
        """Translate a RelationalExpression (=, !=, <, >, <=, >=).

        For SPARQL value equality, numeric types should be compared by value,
        not lexical form. E.g., 1 = 01 is true for integers.
        """
        assert expr.name == "RelationalExpression"
        op = expr.op
        if op not in _RELATIONAL_OPS:
            raise NotImplementedError(f"Relational op {op!r} not implemented")

        left_expr, right_expr = expr.expr, expr.other

        # Normalise: if literal on left and variable on right, swap them
        # (reversing op for ordering comparisons)
        left_type = self._get_type_column(left_expr, var_to_column)
        right_type = self._get_type_column(right_expr, var_to_column)
        if right_type is not None and left_type is None:
            left_expr, right_expr = right_expr, left_expr
            left_type, right_type = right_type, left_type
            op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(op, op)

        left = self._translate_expr(left_expr, var_to_column)
        right = self._translate_expr(right_expr, var_to_column)
        right_literal_type = self._get_literal_type(right_expr)

        # Case 1: variable vs literal (left has type column, right has literal type)
        if left_type is not None and right_literal_type is not None:
            return self._value_aware_comparison_with_literal(
                left, left_type, right, right_literal_type, op
            )

        # Case 2: variable vs variable (both have type columns)
        if left_type is not None and right_type is not None:
            return self._value_aware_comparison(left, left_type, right, right_type, op)

        # Case 3: simple comparison (URIs, etc.)
        return _RELATIONAL_OPS[op](left, right)

    def _get_type_column(self, expr, var_to_column):
        """Get the type column for an expression if it's a variable with type info."""
        if isinstance(expr, Variable):
            type_col_name = f"_ot_{expr}"
            if type_col_name in var_to_column:
                return var_to_column[type_col_name]
        return None

    def _get_literal_type(self, expr):
        """Get the datatype URI string for a Literal expression."""
        if isinstance(expr, Literal):
            if expr.datatype:
                return str(expr.datatype)
            elif expr.language:
                return f"@{expr.language}"
            else:
                return str(XSD.string)
        return None

    def _value_aware_comparison(self, left, left_type, right, right_type, op):
        """Generate SQL for value-aware comparison (handles numeric types).

        When both operands are numeric types, compares numeric values.
        Otherwise compares lexically, but only if types are compatible.
        IRIs and BNodes have NULL type columns, handled via IS NULL checks.
        """
        numeric_types = [literal(t) for t in _NUMERIC_TYPES]
        both_numeric = and_(left_type.in_(numeric_types), right_type.in_(numeric_types))
        lexical_compatible = or_(
            and_(left_type.is_(None), right_type.is_(None)),  # both IRIs/BNodes
            and_(
                left_type.isnot(None), right_type.isnot(None), left_type == right_type
            ),
        )

        left_real, right_real = func.cast(left, Float), func.cast(right, Float)
        cmp_op = _RELATIONAL_OPS[op]

        if op in ("=", "!="):
            equal_cond = or_(
                and_(both_numeric, left_real == right_real),
                and_(lexical_compatible, left == right),
            )
            return equal_cond if op == "=" else not_(equal_cond)
        else:  # <, >, <=, >=
            return case(
                (both_numeric, cmp_op(left_real, right_real)), else_=cmp_op(left, right)
            )

    def _value_aware_comparison_with_literal(
        self, var_value, var_type, lit_value, lit_type_str, op
    ):
        """Compare a variable (with type column) to a literal (with known type).

        SPARQL semantics: incompatible types produce type error (row excluded).
        """
        cmp_op = _RELATIONAL_OPS[op]

        if lit_type_str in _NUMERIC_TYPES:
            # Numeric literal: require var to also be numeric, compare as floats
            var_is_numeric = var_type.in_([literal(t) for t in _NUMERIC_TYPES])
            return and_(
                var_is_numeric,
                cmp_op(func.cast(var_value, Float), func.cast(lit_value, Float)),
            )
        else:
            # Non-numeric literal: require same type, compare lexically
            return and_(var_type == literal(lit_type_str), cmp_op(var_value, lit_value))

    def _translate_conditional_and(self, expr, var_to_column):
        """Translate a ConditionalAndExpression (&&)."""
        assert expr.name == "ConditionalAndExpression"
        operands = [self._translate_expr(expr.expr, var_to_column)]
        operands.extend(self._translate_expr(e, var_to_column) for e in expr.other)
        return and_(*operands)

    def _translate_conditional_or(self, expr, var_to_column):
        """Translate a ConditionalOrExpression (||)."""
        assert expr.name == "ConditionalOrExpression"
        operands = [self._translate_expr(expr.expr, var_to_column)]
        operands.extend(self._translate_expr(e, var_to_column) for e in expr.other)
        return or_(*operands)

    def _translate_binary_chain(self, expr, var_to_column):
        """Translate chained binary expressions (+/-, *//)."""
        assert expr.name in ("AdditiveExpression", "MultiplicativeExpression")
        base = self._translate_expr(expr.expr, var_to_column)
        ops = getattr(expr, "op", [])
        others = getattr(expr, "other", [])
        for op, other in zip(ops, others):
            rhs = self._translate_expr(other, var_to_column)
            if op not in _BINARY_OPS:
                raise NotImplementedError(f"Binary op {op!r} not implemented")
            base = _BINARY_OPS[op](base, rhs)
        return base

    def _translate_in_expression(self, expr, var_to_column):
        """Translate IN / NOT IN expressions."""
        assert expr.name == "InExpression"
        lhs = self._translate_expr(expr.expr, var_to_column)
        items = [self._translate_expr(v, var_to_column) for v in expr.other]
        return lhs.not_in(items) if expr.notin else lhs.in_(items)

    def _translate_builtin(self, expr, var_to_column):
        """Translate SPARQL built-in functions (Builtin_XXX)."""
        fname = expr.name[8:].upper()  # Strip "Builtin_" prefix

        # Get arguments - may be in 'arg' (list or single) or 'args'
        raw_args = getattr(expr, "arg", None) or getattr(expr, "args", [])
        if not isinstance(raw_args, list):
            raw_args = [raw_args]
        args = [self._translate_expr(a, var_to_column) for a in raw_args]

        # --- Simple dispatch functions ---
        if fname in _BUILTIN_SIMPLE:
            return _BUILTIN_SIMPLE[fname](*args)

        # --- No-arg functions ---
        if fname in _BUILTIN_NOARG:
            return _BUILTIN_NOARG[fname]()

        # --- Conditional forms ---
        if fname == "IF":
            return case((args[0], args[1]), else_=args[2])
        if fname == "BOUND":
            return args[0].isnot(None)

        # --- String functions with special handling ---
        if fname == "SUBSTR":
            return self._translate_substr(expr, args, var_to_column)
        if fname == "CONCAT":
            return self._translate_concat(args)
        if fname == "STR":
            return args[0]
        if fname == "STRSTARTS":
            s = self._translate_expr(expr.arg1, var_to_column)
            prefix = self._translate_expr(expr.arg2, var_to_column)
            return s.like(prefix.concat(literal("%")))
        if fname == "STRENDS":
            s = self._translate_expr(expr.arg1, var_to_column)
            suffix = self._translate_expr(expr.arg2, var_to_column)
            return s.like(literal("%").concat(suffix))
        if fname == "CONTAINS":
            s = self._translate_expr(expr.arg1, var_to_column)
            frag = self._translate_expr(expr.arg2, var_to_column)
            return s.like(literal("%").concat(frag).concat(literal("%")))

        # --- Dialect-specific (stubs) ---
        if fname in ("REGEX", "REPLACE"):
            raise NotImplementedError(
                f"{fname} mapping is dialect-specific and not implemented"
            )

        # --- Date/time extraction ---
        if fname in _DATETIME_FIELDS:
            return func.extract(fname.lower(), *args)

        # --- Hash functions ---
        if fname in _HASH_FUNCS:
            return getattr(func, fname.lower())(*args)

        # --- RDF term comparison ---
        if fname == "SAMETERM":
            arg1 = self._translate_expr(expr.arg1, var_to_column)
            arg2 = self._translate_expr(expr.arg2, var_to_column)

            # sameTerm requires both value AND type to match
            # Get type columns for both operands
            type1 = self._get_type_column(expr.arg1, var_to_column)
            type2 = self._get_type_column(expr.arg2, var_to_column)

            if type1 is not None and type2 is not None:
                # Both have type info: require value AND type match
                # NULL types (for IRIs/bnodes) should match with IS NULL
                return and_(
                    arg1 == arg2,
                    or_(
                        and_(type1.is_(None), type2.is_(None)),
                        type1 == type2,
                    ),
                )
            elif type1 is not None or type2 is not None:
                # Mixed: one has type, one doesn't - need careful handling
                # If one is from object column (has type) and one is literal,
                # they can only match if the literal's type matches
                return arg1 == arg2
            else:
                # Neither has type info - simple value comparison
                return arg1 == arg2

        # --- Schema-dependent functions ---
        if fname in _SCHEMA_DEPENDENT_FUNCS:
            raise NotImplementedError(f"{fname} requires schema-specific mapping")

        raise NotImplementedError(
            f"SPARQL built-in function {fname} is not implemented"
        )

    def _translate_substr(self, expr, args, var_to_column):
        """Translate SUBSTR with its special argument structure."""
        # TODO I think the args have already been translated by the time we get here
        string_arg = args[0] if args else self._translate_expr(expr.arg, var_to_column)
        start = (
            self._translate_expr(expr.start, var_to_column)
            if hasattr(expr, "start")
            else None
        )
        length = (
            self._translate_expr(expr.length, var_to_column)
            if hasattr(expr, "length") and expr.length is not None
            else None
        )
        if length is not None:
            return func.substr(string_arg, start, length)
        elif start is not None:
            return func.substr(string_arg, start)
        return func.substr(string_arg)

    def _translate_concat(self, args):
        """Translate CONCAT with chained concatenation."""
        if len(args) == 0:
            return literal("")
        if len(args) == 1:
            return args[0]
        result = args[0]
        for arg in args[1:]:
            result = result.concat(arg)
        return result

    def _translate_aggregate_expr(self, expr, var_to_column):
        """Translate Aggregate_* expressions (COUNT, SUM, AVG etc)."""
        agg_var = getattr(expr, "vars", None)
        arg = self._translate_expr(agg_var, var_to_column) if agg_var else literal(1)
        distinct = bool(getattr(expr, "distinct", None))

        if expr.name in _AGG_FUNCS:
            return _AGG_FUNCS[expr.name](arg.distinct() if distinct else arg)

        if expr.name == "Aggregate_Group_Concat":
            sep = getattr(expr, "separator", ",")
            dialect_funcs = {
                "sqlite": func.group_concat,
                "postgresql": func.string_agg,
                "databricks": func.string_agg,
            }
            f = dialect_funcs.get(self.engine.dialect.name)
            if not f:
                raise NotImplementedError(
                    f"GROUP_CONCAT not supported for {self.engine.dialect.name}"
                )
            return f(arg.distinct(), sep) if distinct else f(arg, sep)

        raise NotImplementedError(f"Aggregate {expr.name!r} not implemented")


# Example usage
if __name__ == "__main__":
    sparql_query = """
    PREFIX dbx: <http://www.databricks.com/ontology/>
    
    SELECT DISTINCT ?tab_prefix
    WHERE {
      ?tab a dbx:Table .
      ?cat a dbx:Catalog .
      ?tab dbx:tableInSchema/dbx:inCatalog ?cat .
      BIND(SUBSTR(STR(?tab), 1, 10) AS ?tab_prefix)
    }
    """

    engine = create_databricks_engine(
        server_hostname="e2-demo-field-eng.cloud.databricks.com",
        http_path="/sql/1.0/warehouses/8baced1ff014912d",
        access_token="",
    )

    translator = AlgebraTranslator(
        engine=engine, table_name="users.joshua_green.triples"
    )

    res = translator.execute(sparql_query)
    print("Results:", res.fetchall())

    engine.dispose()
