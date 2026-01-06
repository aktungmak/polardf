import itertools
import warnings
from typing import List, Optional, Union

from rdflib import Literal, URIRef
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
)
from sqlalchemy.sql import column


def term_to_string(term) -> Optional[str]:
    """Convert an RDF term (URIRef, Literal, or BNode) to its string representation.

    Returns None if the term is not a recognized RDF term type (e.g., a Variable).

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
    - Datatype URI string for typed literals
    - Empty string "" for plain literals (no datatype, no language)
    """
    if isinstance(term, Literal):
        if term.datatype:
            return str(term.datatype)
        elif term.language:
            return f"@{term.language}"
        else:
            # TODO should we set a default value for plain literals?
            return ""  # Plain literal or BNode
    return None  # URIRef, BNode, or unknown


def create_databricks_engine(
    server_hostname: str, http_path: str, access_token: str, **engine_kwargs
) -> Engine:
    engine_uri = (
        f"databricks://token:{access_token}@{server_hostname}?http_path={http_path}"
    )
    return create_engine(engine_uri, **engine_kwargs)


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
            # Also filter by object type
            o_type = term_to_object_type(o)
            if o_type is None:
                # URIRef or BNode - ot should be NULL
                conditions.append(self.table.c.ot.is_(None))
            else:
                # Literal - ot should match the type
                conditions.append(self.table.c.ot == o_type)
        else:
            raise NotImplementedError(f"Object type {type(o)} not implemented")

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
            # Also filter by object type
            if o_type is None:
                # URIRef or BNode - ot should be NULL
                conditions.append(path_cte.c.ot.is_(None))
            else:
                # Literal - ot should match the type
                conditions.append(path_cte.c.ot == o_type)

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

        # Get column names from base query
        if isinstance(base_query, CTE):
            base_cols = set(base_query.c.keys())
        elif hasattr(base_query, "selected_columns"):
            base_cols = set(col.key for col in base_query.selected_columns)
        else:
            base_cols = set()

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

        # Convert to subquery so we can add columns
        if isinstance(inner_query, CTE):
            subquery = inner_query
        else:
            subquery = inner_query.subquery()

        # Build a context mapping variable names to columns from the subquery
        ctx = {name: subquery.c[name] for name in subquery.c.keys()}

        # Translate the expression
        sql_expr = self._translate_expr(expr, ctx)

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

        # Build context from available columns
        if isinstance(inner_query, CTE):
            ctx = {name: inner_query.c[name] for name in inner_query.c.keys()}
            base_select = select(*inner_query.c).select_from(inner_query)
        else:
            # It's a Select - get columns and add WHERE directly
            ctx = {col.key: col for col in inner_query.selected_columns}
            base_select = inner_query

        # Translate the filter expression
        sql_condition = self._translate_expr(filter_expr, ctx)

        # Add WHERE clause directly (no extra subquery)
        return base_select.where(sql_condition)

    def _translate_left_join(self, left_join):
        """Translate a LeftJoin (OPTIONAL) operation."""
        left_pattern = left_join["p1"]
        right_pattern = left_join["p2"]
        # expr = left_join["expr"]  # Join condition - TODO: handle filters

        # Translate both sides
        left_query = self._translate_pattern(left_pattern)
        right_query = self._translate_pattern(right_pattern)

        # Convert to CTEs for joining if they aren't already
        left_cte = left_query if isinstance(left_query, CTE) else left_query.cte()
        right_cte = right_query if isinstance(right_query, CTE) else right_query.cte()

        # Find common variables for join condition
        common_cols = set(left_cte.c.keys()) & set(right_cte.c.keys())

        if common_cols:
            join_conditions = [
                left_cte.c[col] == right_cte.c[col] for col in common_cols
            ]

            # All columns from both sides (deduplicated - common cols only from left)
            all_columns = list(left_cte.c) + [
                col for name, col in right_cte.c.items() if name not in common_cols
            ]

            # LEFT JOIN - return Select (caller can wrap in CTE if needed)
            return select(*all_columns).select_from(
                left_cte.outerjoin(right_cte, and_(*join_conditions))
            )
        else:
            # If no common variables, it's a cartesian product with optional semantics
            all_columns = list(left_cte.c) + list(right_cte.c)
            return select(*all_columns).select_from(
                left_cte.outerjoin(right_cte, true())
            )

    def _translate_expr(self, expr, ctx):
        """Translate an rdflib SPARQL algebra expression (CompValue / term)
        into a SQLAlchemy expression.

        Args:
            expr: The expression to translate (Variable, Literal, or CompValue)
            ctx: A dict mapping variable names to SQLAlchemy columns
        """
        # Base case: Variable - look up in context
        if isinstance(expr, Variable):
            var_name = str(expr)
            if var_name in ctx:
                return ctx[var_name]
            raise ValueError(f"Variable ?{var_name} not found in context")

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

        # ---------- Logical & comparison ----------
        if name == "RelationalExpression":
            left = self._translate_expr(expr.expr, ctx)
            right = self._translate_expr(expr.other, ctx)
            op = expr.op
            if op == "=":
                return left == right
            if op == "!=":
                return left != right
            if op == "<":
                return left < right
            if op == ">":
                return left > right
            if op == "<=":
                return left <= right
            if op == ">=":
                return left >= right
            raise NotImplementedError(f"Relational op {op!r} not supported")

        if name == "ConditionalAndExpression":
            operands = [self._translate_expr(expr.expr, ctx)]
            operands.extend(self._translate_expr(e, ctx) for e in expr.other)
            return and_(*operands)
        if name == "ConditionalOrExpression":
            operands = [self._translate_expr(expr.expr, ctx)]
            operands.extend(self._translate_expr(e, ctx) for e in expr.other)
            return or_(*operands)
        if name == "UnaryNot":
            return not_(self._translate_expr(expr.expr, ctx))

        # ---------- Arithmetic ----------
        if name == "AdditiveExpression":
            base = self._translate_expr(expr.expr, ctx)
            ops = expr.op if hasattr(expr, "op") else []
            others = expr.other if hasattr(expr, "other") else []
            for op, other in zip(ops, others):
                rhs = self._translate_expr(other, ctx)
                if op == "+":
                    base = base + rhs
                elif op == "-":
                    base = base - rhs
                else:
                    raise NotImplementedError(f"Additive op {op!r} not supported")
            return base

        if name == "MultiplicativeExpression":
            base = self._translate_expr(expr.expr, ctx)
            ops = expr.op if hasattr(expr, "op") else []
            others = expr.other if hasattr(expr, "other") else []
            for op, other in zip(ops, others):
                rhs = self._translate_expr(other, ctx)
                if op == "*":
                    base = base * rhs
                elif op == "/":
                    base = base / rhs
                else:
                    raise NotImplementedError(f"Multiplicative op {op!r} not supported")
            return base

        # ---------- IN / NOT IN ----------
        if name == "InExpression":
            lhs = self._translate_expr(expr.expr, ctx)
            items = [self._translate_expr(v, ctx) for v in expr.other]
            return lhs.not_in(items) if expr.notin else lhs.in_(items)

        # ---------- Built-in functions (Builtin_XXX pattern) ----------
        if name.startswith("Builtin_"):
            fname = name[8:].upper()  # Strip "Builtin_" prefix

            # Get arguments - may be in 'arg' (list or single) or 'args'
            raw_args = getattr(expr, "arg", None) or getattr(expr, "args", [])
            if not isinstance(raw_args, list):
                raw_args = [raw_args]
            args = [self._translate_expr(a, ctx) for a in raw_args]

            # --- Conditional forms ---
            if fname == "IF":
                cond = args[0]
                then = args[1]
                els = args[2]
                return case((cond, then), else_=els)
            if fname == "COALESCE":
                return func.coalesce(*args)
            if fname == "BOUND":
                # BOUND(?var) tests if variable is not NULL
                return args[0].isnot(None)

            # --- String functions ---
            if fname == "STRLEN":
                return func.length(*args)
            if fname == "SUBSTR":
                # SPARQL SUBSTR has special argument structure: arg, start, length
                # The arg is in args[0], but start/length are separate attributes
                string_arg = args[0] if args else self._translate_expr(expr.arg, ctx)
                start = (
                    self._translate_expr(expr.start, ctx)
                    if hasattr(expr, "start")
                    else None
                )
                length = (
                    self._translate_expr(expr.length, ctx)
                    if hasattr(expr, "length") and expr.length is not None
                    else None
                )
                if length is not None:
                    return func.substr(string_arg, start, length)
                elif start is not None:
                    return func.substr(string_arg, start)
                return func.substr(string_arg)
            if fname == "UCASE":
                return func.upper(*args)
            if fname == "LCASE":
                return func.lower(*args)
            if fname == "CONCAT":
                # SQLite uses || for concat, but func.concat works for most DBs
                # For broader compatibility, we can use explicit concatenation
                if len(args) == 0:
                    return literal("")
                if len(args) == 1:
                    return args[0]
                # Chain concatenation with ||
                result = args[0]
                for arg in args[1:]:
                    result = result.concat(arg)
                return result
            if fname == "STR":
                # STR just returns the lexical form - identity for string storage
                return args[0]

            if fname == "STRSTARTS":
                s = self._translate_expr(expr.arg1, ctx)
                prefix = self._translate_expr(expr.arg2, ctx)
                return s.like(prefix.concat(literal("%")))
            if fname == "STRENDS":
                s = self._translate_expr(expr.arg1, ctx)
                suffix = self._translate_expr(expr.arg2, ctx)
                return s.like(literal("%").concat(suffix))
            if fname == "CONTAINS":
                s = self._translate_expr(expr.arg1, ctx)
                frag = self._translate_expr(expr.arg2, ctx)
                return s.like(literal("%").concat(frag).concat(literal("%")))

            # REGEX/REPLACE would be dialect-specific; stub:
            if fname == "REGEX":
                raise NotImplementedError("REGEX mapping is dialect-specific")
            if fname == "REPLACE":
                raise NotImplementedError("REPLACE mapping is dialect-specific")

            # --- Numeric ---
            if fname == "ABS":
                return func.abs(*args)
            if fname == "ROUND":
                return func.round(*args)
            if fname == "CEIL":
                return func.ceil(*args)
            if fname == "FLOOR":
                return func.floor(*args)
            if fname == "RAND":
                return func.random()  # adjust to func.rand() if needed

            # --- Date/time ---
            if fname == "NOW":
                return func.current_timestamp()
            if fname in ("YEAR", "MONTH", "DAY", "HOURS", "MINUTES", "SECONDS"):
                field = fname.lower()
                return func.extract(field, *args)

            # --- Hashes ---
            if fname == "MD5":
                return func.md5(*args)
            if fname in ("SHA1", "SHA256", "SHA384", "SHA512"):
                return getattr(func, fname.lower())(*args)

            # --- RDF term comparison ---
            if fname == "SAMETERM":
                # sameTerm(?a, ?b) tests if two RDF terms are identical
                arg1 = self._translate_expr(expr.arg1, ctx)
                arg2 = self._translate_expr(expr.arg2, ctx)
                return arg1 == arg2

            # --- RDF term helpers (schema-dependent) ---
            if fname in (
                "LANG",
                "DATATYPE",
                "ISIRI",
                "ISBLANK",
                "ISLITERAL",
                "ISNUMERIC",
            ):
                raise NotImplementedError(f"{fname} requires schema-specific mapping")

            raise NotImplementedError(
                f"SPARQL built-in function {fname} is not supported"
            )

        # ---------- Aggregates as expressions ----------
        if name == "Aggregate":
            agg_name = expr.AggFunc.upper()
            agg_vars = getattr(expr, "vars", None)
            if agg_vars:
                arg = self._translate_expr(agg_vars[0], ctx)
            else:
                arg = literal(1)  # COUNT(*) case
            distinct = getattr(expr, "distinct", False)

            if agg_name == "COUNT":
                return func.count(arg.distinct()) if distinct else func.count(arg)
            if agg_name == "SUM":
                f = func.sum
            elif agg_name == "AVG":
                f = func.avg
            elif agg_name == "MIN":
                f = func.min
            elif agg_name == "MAX":
                f = func.max
            elif agg_name == "SAMPLE":
                warnings.warn("SAMPLE aggregate is not supported, falling back to MIN")
                f = func.min
            elif agg_name == "GROUP_CONCAT":
                if self.engine.dialect.name == "sqlite":
                    f = func.group_concat
                elif self.engine.dialect.name in ["postgresql", "databricks"]:
                    f = func.string_agg
                else:
                    raise NotImplementedError(
                        f"GROUP_CONCAT aggregate is not supported for dialect {self.engine.dialect.name}"
                    )
                sep = getattr(expr, "separator", ",")
                if distinct:
                    return f(arg.distinct(), sep)
                return f(arg, sep)
            else:
                raise NotImplementedError(f"Aggregate {agg_name!r} not supported")

            return f(arg.distinct() if distinct else arg)

        raise NotImplementedError(f"Expression kind {name!r} not handled by translator")


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
