import itertools
import warnings
from typing import Dict, List, Optional, Union

from rdflib import Literal, URIRef, XSD
from rdflib.paths import Path, SequencePath, MulPath
from rdflib.plugins.sparql import parser, algebra
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import Variable
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
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
)
from sqlalchemy.sql import column


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
    return None  # URIRef, BNode, or unknown


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
_HASH_FUNCS = {"SHA1", "SHA256", "SHA384", "SHA512"}

# Date/time extraction fields
_DATETIME_FIELDS = {"YEAR", "MONTH", "DAY", "HOURS", "MINUTES", "SECONDS"}

# Aggregate functions that map directly
_AGG_FUNCS = {
    "SUM": func.sum,
    "AVG": func.avg,
    "MIN": func.min,
    "MAX": func.max,
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


class AlgebraTranslator:
    """Translates SPARQL algebra expressions to SQLAlchemy queries.
    The assumption is that the triples are stored in a table with columns s, p, and o.
    """

    def __init__(
        self, engine: Engine, table_name: str = "triples", create_table: bool = False
    ):
        self.engine = engine
        self.metadata = MetaData()
        self._cte_counter = itertools.count()

        # Define the triple table structure
        self.table = Table(
            quoted_name(table_name, quote=False),
            self.metadata,
            Column("s", String, nullable=False, primary_key=True),
            Column("p", String, nullable=False, primary_key=True),
            Column("o", String, nullable=False, primary_key=True),
            Column("ot", String, nullable=True, primary_key=True),
        )

        # TODO create a symbol table alterantive like this
        # SELECT t.* FROM t INNER JOIN s s1 ON t.s = s1.i AND s1.t = "one" INNER JOIN s s2 ON t.o = s2.i AND s2.t = "three";

        if create_table:
            self.metadata.create_all(self.engine)

    # ---------- Helper Methods ----------

    def _add_object_type_condition(
        self, conditions: list, ot_column, o_type: Optional[str]
    ):
        """Add object type filtering condition to the conditions list."""
        if o_type is None:
            conditions.append(ot_column.is_(None))
        else:
            conditions.append(ot_column == o_type)

    def _ensure_subquery(self, query):
        """Convert query to a subquery/CTE for column access."""
        return query if isinstance(query, CTE) else query.subquery()

    def _get_var_to_column(self, query) -> Dict[str, Column]:
        """Build a variable->column mapping from query."""
        if isinstance(query, CTE):
            return {name: query.c[name] for name in query.c.keys()}
        elif hasattr(query, "selected_columns"):
            return {col.key: col for col in query.selected_columns}
        return {}

    def _get_column_names(self, query) -> set:
        """Get the set of column names from a query."""
        if isinstance(query, CTE):
            return set(query.c.keys())
        elif hasattr(query, "selected_columns"):
            return set(col.key for col in query.selected_columns)
        return set()

    # ---------- Core Translation Methods ----------

    def execute(self, sparql_query: str) -> CursorResult:
        """Translate a SPARQL query and execute it."""
        sql_query = self.translate(sparql_query)
        with self.engine.connect() as conn:
            return conn.execute(sql_query)

    def translate(self, sparql_string: str):
        """Translate a SPARQL query string to a SQLAlchemy query."""
        query_tree = parser.parseQuery(sparql_string)
        query_algebra = algebra.translateQuery(query_tree).algebra
        if hasattr(query_algebra, "name"):
            if query_algebra.name == "SelectQuery":
                return self._translate_select_query(query_algebra)

        raise NotImplementedError(
            f"Algebra translation not implemented for {query_algebra}"
        )

    def _translate_select_query(self, select_query: CompValue):
        base_query = self._translate_pattern(select_query["p"])

        # If the base query is a CTE, we need to select from it
        if isinstance(base_query, CTE):
            return select(*base_query.c).select_from(base_query)

        # Otherwise, the base query is already a complete select, just return it
        return base_query

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
            else:
                raise NotImplementedError(
                    f"Pattern type {pattern.name} not implemented"
                )
        else:
            raise ValueError(f"Unknown pattern type: {pattern}")

    def _translate_bgp(self, bgp: CompValue) -> Union[Select, CTE]:
        """Translate a Basic Graph Pattern to a query."""
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

    def _triple_to_query(self, triple) -> Select:
        """Convert a single triple pattern to a query."""
        s, p, o = triple

        # Build SELECT columns and WHERE conditions
        columns = []
        conditions = []

        # Subject
        if isinstance(s, Variable):
            columns.append(self.table.c.s.label(str(s)))
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
        elif (p_str := term_to_string(p)) is not None:
            conditions.append(self.table.c.p == p_str)
        else:
            raise NotImplementedError(f"Predicate type {type(p)} not implemented")

        # Object
        if isinstance(o, Variable):
            if o == s:  # Same variable as subject
                conditions.append(self.table.c.o == self.table.c.s)
            elif o == p:  # Same variable as predicate
                conditions.append(self.table.c.o == self.table.c.p)
            else:
                columns.append(self.table.c.o.label(str(o)))
        elif (o_str := term_to_string(o)) is not None:
            conditions.append(self.table.c.o == o_str)
            o_type = term_to_object_type(o)
            self._add_object_type_condition(conditions, self.table.c.ot, o_type)
        else:
            raise NotImplementedError(f"Object type {type(o)} not implemented")

        # If no variables to select, we still need a valid SELECT clause
        # This happens when all triple positions are bound (existence check)
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
            self._add_object_type_condition(conditions, path_cte.c.ot, o_type)

        if conditions:
            final_query = final_query.where(and_(*conditions))

        return final_query

    def _join_queries(self, queries):
        """Natural join multiple queries on common variables."""
        # no-op if only one argument
        if len(queries) == 1:
            return queries[0]

        # Convert all queries to CTEs
        ctes = [query if isinstance(query, CTE) else query.cte() for query in queries]

        # Generate join conditions for all pairs of CTEs based on common columns
        conditions = []
        for left, right in itertools.combinations(ctes, 2):
            common_cols = set(left.c.keys()) & set(right.c.keys())
            for col in common_cols:
                conditions.append(left.c[col] == right.c[col])

        # Build column list: include each column name only once (avoid duplicates from joins)
        projection_cols = {}
        for cte in ctes:
            for name, col in cte.c.items():
                projection_cols.setdefault(name, col)

        # Select deduplicated columns from all CTEs with WHERE conditions
        return select(*projection_cols.values()).select_from(*ctes).where(*conditions)

    def _translate_project(self, project):
        """Translate a Project (SELECT) operation."""
        project_vars = project["PV"]
        pattern = project["p"]

        # Translate the inner pattern
        base_query = self._translate_pattern(pattern)
        base_cols = self._get_column_names(base_query)
        project_var_names = set(str(var) for var in project_vars)

        if base_cols == project_var_names:
            # Projection doesn't change anything, return base query
            return base_query

        # Handle empty projections (e.g., when all terms are bound)
        if len(project_vars) == 0:
            # For empty projections, return base query as-is
            # This allows MulPath queries with bound endpoints to still work
            return base_query

        # Build projection columns
        # - Variables that exist in base query: select from subquery
        # - Variables that don't exist: add as NULL (unbound in SPARQL)
        subquery = base_query.subquery()
        var_columns = []
        for var in project_vars:
            var_name = str(var)
            if var_name in base_cols:
                var_columns.append(column(var_name))
            else:
                var_columns.append(null().label(var_name))

        return select(*var_columns).select_from(subquery)

    def _translate_distinct(self, distinct):
        """Translate a Distinct operation."""
        inner_query = self._translate_pattern(distinct["p"])

        # If it's a CTE, select from it with distinct
        if isinstance(inner_query, CTE):
            return select(*inner_query.c).select_from(inner_query).distinct()

        # Otherwise apply distinct directly
        return inner_query.distinct()

    def _translate_extend(self, extend):
        """Translate an Extend (BIND) operation.

        The Extend node adds a new variable binding computed from an expression.
        Structure: Extend(p=inner_pattern, var=new_variable_name, expr=expression)
        """
        inner_pattern = extend["p"]
        var_name = str(extend["var"])
        expr = extend["expr"]

        # Translate the inner pattern
        inner_query = self._translate_pattern(inner_pattern)
        subquery = self._ensure_subquery(inner_query)
        var_to_column = self._get_var_to_column(subquery)

        # Translate the expression
        sql_expr = self._translate_expr(expr, var_to_column)

        # Build result columns: all existing columns plus the new computed column
        result_columns = list(subquery.c) + [sql_expr.label(var_name)]

        return select(*result_columns).select_from(subquery)

    def _translate_filter(self, filter_node):
        """Translate a Filter operation.

        Filter applies a boolean expression to filter results from the inner pattern.
        Structure: Filter(p=inner_pattern, expr=boolean_expression)
        """
        inner_pattern = filter_node["p"]
        filter_expr = filter_node["expr"]

        # Translate the inner pattern
        inner_query = self._translate_pattern(inner_pattern)

        # Build var_to_column from available columns
        if isinstance(inner_query, CTE):
            var_to_column = self._get_var_to_column(inner_query)
            base_select = select(*inner_query.c).select_from(inner_query)
        else:
            var_to_column = self._get_var_to_column(inner_query)
            base_select = inner_query

        # Translate the filter expression
        sql_condition = self._translate_expr(filter_expr, var_to_column)

        # Add WHERE clause directly (no extra subquery)
        return base_select.where(sql_condition)

    def _translate_order_by(self, order_by):
        """Translate an OrderBy operation.

        OrderBy applies ordering to results from the inner pattern.
        Structure: OrderBy(p=inner_pattern, expr=list_of_order_conditions)
        """
        inner_pattern = order_by["p"]
        order_conditions = order_by["expr"]

        # Translate the inner pattern
        inner_query = self._translate_pattern(inner_pattern)

        # Build var_to_column for expression translation
        if isinstance(inner_query, CTE):
            var_to_column = self._get_var_to_column(inner_query)
            base_select = select(*inner_query.c).select_from(inner_query)
        else:
            var_to_column = self._get_var_to_column(inner_query)
            base_select = inner_query

        # Build ORDER BY clauses
        order_clauses = []
        for cond in order_conditions:
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

    def _translate_join(self, join):
        """Translate a Join operation (natural join of two patterns).

        Join combines two graph patterns using natural join semantics -
        matching rows on common variables.
        Structure: Join(p1=left_pattern, p2=right_pattern)
        """
        left_pattern = join["p1"]
        right_pattern = join["p2"]

        # Translate both sides
        left_query = self._translate_pattern(left_pattern)
        right_query = self._translate_pattern(right_pattern)

        # Reuse _join_queries which handles natural join on common columns
        return self._join_queries([left_query, right_query])

    def _translate_left_join(self, left_join):
        """Translate a LeftJoin (OPTIONAL) operation.

        In SPARQL, OPTIONAL { pattern FILTER(expr) } becomes:
        LeftJoin(p1, p2, expr) where expr is the filter condition.

        The expr is applied as part of the ON condition of the LEFT JOIN,
        NOT as a WHERE clause (which would filter out rows that don't match).
        """
        left_pattern = left_join["p1"]
        right_pattern = left_join["p2"]
        filter_expr = left_join.get("expr")

        # Translate both sides
        left_query = self._translate_pattern(left_pattern)
        right_query = self._translate_pattern(right_pattern)

        # Convert to CTEs for joining if they aren't already
        left_cte = left_query if isinstance(left_query, CTE) else left_query.cte()
        right_cte = right_query if isinstance(right_query, CTE) else right_query.cte()

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

    # ---------- Expression Translation ----------

    def _translate_expr(self, expr, var_to_column):
        """Translate an rdflib SPARQL algebra expression (CompValue / term)
        into a SQLAlchemy expression.

        Args:
            expr: The expression to translate (Variable, Literal, or CompValue)
            var_to_column: A dict mapping variable names to SQLAlchemy columns
        """
        # Base case: Variable - look up in var_to_column
        if isinstance(expr, Variable):
            var_name = str(expr)
            if var_name in var_to_column:
                return var_to_column[var_name]
            # Unbound variable - return NULL to use SQL's NULL comparison semantics
            # In SPARQL, comparisons with unbound variables produce errors, which are
            # treated as FALSE in filter contexts. SQL's NULL comparison semantics
            # achieve the same result: NULL = x evaluates to UNKNOWN, filtered out.
            return null()

        # Base case: Literal or URIRef - convert to SQL literal string
        if isinstance(expr, (Literal, URIRef)):
            # Handle boolean literals specially for FILTER expressions
            if isinstance(expr, Literal) and expr.datatype == URIRef(
                "http://www.w3.org/2001/XMLSchema#boolean"
            ):
                return true() if expr.toPython() else not_(true())
            return literal(str(expr))

        if not isinstance(expr, CompValue):
            # raw constant or already an expression
            return literal(expr)

        name = expr.name

        # Dispatch to specific handlers
        if name == "RelationalExpression":
            return self._translate_relational(expr, var_to_column)
        if name == "ConditionalAndExpression":
            return self._translate_conditional_and(expr, var_to_column)
        if name == "ConditionalOrExpression":
            return self._translate_conditional_or(expr, var_to_column)
        if name == "UnaryNot":
            return not_(self._translate_expr(expr.expr, var_to_column))
        if name in ("AdditiveExpression", "MultiplicativeExpression"):
            return self._translate_binary_chain(expr, var_to_column)
        if name == "InExpression":
            return self._translate_in_expression(expr, var_to_column)
        if name.startswith("Builtin_"):
            return self._translate_builtin(expr, var_to_column)
        if name == "Aggregate":
            return self._translate_aggregate(expr, var_to_column)

        raise NotImplementedError(f"Expression kind {name!r} not handled by translator")

    def _translate_relational(self, expr, var_to_column):
        """Translate a RelationalExpression (=, !=, <, >, <=, >=)."""
        left = self._translate_expr(expr.expr, var_to_column)
        right = self._translate_expr(expr.other, var_to_column)
        op = expr.op
        if op in _RELATIONAL_OPS:
            return _RELATIONAL_OPS[op](left, right)
        raise NotImplementedError(f"Relational op {op!r} not supported")

    def _translate_conditional_and(self, expr, var_to_column):
        """Translate a ConditionalAndExpression (&&)."""
        operands = [self._translate_expr(expr.expr, var_to_column)]
        operands.extend(self._translate_expr(e, var_to_column) for e in expr.other)
        return and_(*operands)

    def _translate_conditional_or(self, expr, var_to_column):
        """Translate a ConditionalOrExpression (||)."""
        operands = [self._translate_expr(expr.expr, var_to_column)]
        operands.extend(self._translate_expr(e, var_to_column) for e in expr.other)
        return or_(*operands)

    def _translate_binary_chain(self, expr, var_to_column):
        """Translate chained binary expressions (+/-, *//)."""
        base = self._translate_expr(expr.expr, var_to_column)
        ops = getattr(expr, "op", [])
        others = getattr(expr, "other", [])
        for op, other in zip(ops, others):
            rhs = self._translate_expr(other, var_to_column)
            if op not in _BINARY_OPS:
                raise NotImplementedError(f"Binary op {op!r} not supported")
            base = _BINARY_OPS[op](base, rhs)
        return base

    def _translate_in_expression(self, expr, var_to_column):
        """Translate IN / NOT IN expressions."""
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
            raise NotImplementedError(f"{fname} mapping is dialect-specific")

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
            return arg1 == arg2

        # --- Schema-dependent functions ---
        if fname in _SCHEMA_DEPENDENT_FUNCS:
            raise NotImplementedError(f"{fname} requires schema-specific mapping")

        raise NotImplementedError(f"SPARQL built-in function {fname} is not supported")

    def _translate_substr(self, expr, args, var_to_column):
        """Translate SUBSTR with its special argument structure."""
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

    def _translate_aggregate(self, expr, var_to_column):
        """Translate aggregate expressions (COUNT, SUM, AVG, etc.)."""
        agg_name = expr.AggFunc.upper()
        agg_vars = getattr(expr, "vars", None)
        arg = (
            self._translate_expr(agg_vars[0], var_to_column) if agg_vars else literal(1)
        )
        distinct = getattr(expr, "distinct", False)

        # COUNT has special handling for distinct
        if agg_name == "COUNT":
            return func.count(arg.distinct()) if distinct else func.count(arg)

        # Simple aggregate functions
        if agg_name in _AGG_FUNCS:
            f = _AGG_FUNCS[agg_name]
            return f(arg.distinct() if distinct else arg)

        # SAMPLE falls back to MIN with warning
        if agg_name == "SAMPLE":
            warnings.warn("SAMPLE aggregate is not supported, falling back to MIN")
            return func.min(arg.distinct() if distinct else arg)

        # GROUP_CONCAT is dialect-specific
        if agg_name == "GROUP_CONCAT":
            return self._translate_group_concat(arg, expr, distinct)

        raise NotImplementedError(f"Aggregate {agg_name!r} not supported")

    def _translate_group_concat(self, arg, expr, distinct):
        """Translate GROUP_CONCAT with dialect-specific handling."""
        dialect = self.engine.dialect.name
        if dialect == "sqlite":
            f = func.group_concat
        elif dialect in ("postgresql", "databricks"):
            f = func.string_agg
        else:
            raise NotImplementedError(
                f"GROUP_CONCAT aggregate is not supported for dialect {dialect}"
            )
        sep = getattr(expr, "separator", ",")
        return f(arg.distinct(), sep) if distinct else f(arg, sep)


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

    result = translator.execute(sparql_query)
    print("Results:", result.fetchall())

    engine.dispose()
