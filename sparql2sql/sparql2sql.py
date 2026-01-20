"""SPARQL to SQL translation using SQLAlchemy.

This module translates SPARQL algebra (from rdflib) to SQL queries.
Design principles:
- Explicit context passing (no mutable instance state)
- Unified intermediate representation (always use CTEs)
- Registry-based dispatch (declarative handler registration)
- Separation of concerns (schema, translation, execution are distinct)
"""

import itertools
from dataclasses import dataclass, replace, field
from typing import Callable, Dict, List, Optional, Union

from rdflib import BNode, Literal, URIRef, XSD
from rdflib.paths import MulPath, Path, SequencePath
from rdflib.plugins.sparql import algebra, parser
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import Variable
from sqlalchemy import (
    CTE,
    Column,
    CursorResult,
    Engine,
    Float,
    MetaData,
    Select,
    String,
    Table,
    and_,
    asc,
    case,
    create_engine,
    desc,
    exists,
    func,
    literal,
    not_,
    null,
    or_,
    quoted_name,
    select,
    true,
)


# =============================================================================
# Term Conversion Utilities
# =============================================================================


def term_to_string(term) -> Optional[str]:
    """Convert an RDF term to its string representation for storage.

    Returns None if the term is not a recognised RDF term type (e.g., a Variable).

    The format matches how terms are stored in the database:
    - URIRefs: the raw URI string
    - Literals: the lexical value (type info is stored in the 'ot' column)
    - BNodes: _:id
    """
    if isinstance(term, URIRef):
        return str(term)
    if isinstance(term, Literal):
        return str(term)
    if isinstance(term, BNode):
        return f"_:{term}"
    return None


def term_to_object_type(term) -> Optional[str]:
    """Get the object type for the 'ot' column.

    Returns:
    - None for URIRef and BNode (IRI/blank node)
    - Datatype URI string for typed literals
    - "@lang" for language-tagged literals
    - xsd:string for plain literals (per RDF 1.1)
    """
    if isinstance(term, Literal):
        if term.datatype:
            return str(term.datatype)
        if term.language:
            return f"@{term.language}"
        return str(XSD.string)
    return None


# =============================================================================
# Translation Context
# =============================================================================


@dataclass(frozen=True)
class Context:
    """Immutable translation context passed to all handlers."""

    table: Table
    graph_aware: bool
    graph_term: Optional[Union[Variable, URIRef]] = None
    _cte_count: int = field(default=0, compare=False)

    def with_graph(self, term) -> "Context":
        """Return new context with graph term set."""
        return replace(self, graph_term=term)

    def next_cte_name(self) -> tuple["Context", str]:
        """Return (new_ctx, cte_name) with incremented counter."""
        return replace(self, _cte_count=self._cte_count + 1), f"cte_{self._cte_count}"

    @property
    def graph_filter(self):
        """SQL condition for current graph context, or None."""
        if not self.graph_aware:
            return None
        if self.graph_term is None:
            return self.table.c.g.is_(None)
        if isinstance(self.graph_term, Variable):
            return self.table.c.g.isnot(None)
        return self.table.c.g == str(self.graph_term)


# =============================================================================
# Query Utilities
# =============================================================================


QueryResult = Union[Select, CTE]


def as_cte(query: QueryResult, name: str = None) -> CTE:
    """Normalise query to CTE for uniform column access."""
    return query if isinstance(query, CTE) else query.cte(name)


def cols(query: QueryResult) -> Dict[str, Column]:
    """Get column name -> Column mapping from any query type."""
    if isinstance(query, CTE):
        return {name: query.c[name] for name in query.c.keys()}
    if hasattr(query, "selected_columns"):
        return {col.key: col for col in query.selected_columns}
    return {}


def col_names(query: QueryResult) -> set:
    """Get set of column names from a query."""
    return set(cols(query).keys())


# =============================================================================
# Lookup Tables for Operators and Functions
# =============================================================================

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

_BUILTIN_NOARG = {
    "RAND": func.random,
    "NOW": func.current_timestamp,
}

_HASH_FUNCS = {"SHA1", "SHA256", "SHA384", "SHA512"}

_DATETIME_FIELDS = {"YEAR", "MONTH", "DAY", "HOURS", "MINUTES", "SECONDS"}

_AGG_FUNCS = {
    "Aggregate_Count": func.count,
    "Aggregate_Sum": func.sum,
    "Aggregate_Avg": func.avg,
    "Aggregate_Min": func.min,
    "Aggregate_Max": func.max,
    "Aggregate_Sample": func.min,
}

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

# =============================================================================
# Effective Boolean Value (EBV)
# =============================================================================

# Valid EBV types: numerics, boolean, and string (plus plain literals handled separately)
_EBV_VALID_TYPES = _NUMERIC_TYPES | {str(XSD.boolean), str(XSD.string)}


def _ebv(value_col, type_col):
    """Compute Effective Boolean Value per SPARQL 17.2.2.

    Returns NULL for type errors (unbound/unsupported), which filters the row.
    """
    is_plain = type_col.is_(None)
    valid_types = [literal(t) for t in _EBV_VALID_TYPES]
    numeric_types = [literal(t) for t in _NUMERIC_TYPES]

    return case(
        # Unbound
        (value_col.is_(None), null()),
        # Invalid type
        (not_(or_(is_plain, type_col.in_(valid_types))), null()),
        # Boolean
        (type_col == literal(str(XSD.boolean)), value_col == literal("true")),
        # Numeric
        (type_col.in_(numeric_types), func.cast(value_col, Float) != literal(0)),
        # String
        (
            or_(is_plain, type_col == literal(str(XSD.string))),
            func.length(value_col) > literal(0),
        ),
        # Unreachable
        else_=null(),
    )


def _needs_ebv(expr, var_to_col):
    """Check if an expression is a variable that needs EBV conversion.

    Returns (value_col, type_col) if EBV needed, None otherwise.
    """
    if not isinstance(expr, Variable):
        return None
    var_name = str(expr)
    type_col_name = f"_ot_{var_name}"
    if var_name in var_to_col and type_col_name in var_to_col:
        return var_to_col[var_name], var_to_col[type_col_name]
    return None


# =============================================================================
# Expression Translation Registry
# =============================================================================

_EXPRS: Dict[str, Callable] = {}


def expr_handler(name: str):
    """Decorator to register an expression handler."""

    def decorator(fn):
        _EXPRS[name] = fn
        return fn

    return decorator


def translate_expr(expr, var_to_col: Dict[str, Column], engine: Engine = None):
    """Translate an rdflib SPARQL expression to SQLAlchemy."""
    # Variable lookup
    if isinstance(expr, Variable):
        return var_to_col.get(str(expr), null())

    # Literal values
    if isinstance(expr, Literal):
        if expr.datatype == XSD.boolean:
            return true() if expr.toPython() else not_(true())
        if expr.datatype and str(expr.datatype) in _NUMERIC_TYPES:
            return literal(expr.toPython())
        return literal(str(expr))

    if isinstance(expr, URIRef):
        return literal(str(expr))

    if not isinstance(expr, CompValue):
        return literal(expr)

    # Dispatch to registered handler
    handler = _EXPRS.get(expr.name)
    if handler:
        return handler(expr, var_to_col, engine)

    raise NotImplementedError(f"Expression {expr.name!r} not implemented")


# --- Expression Handlers ---


@expr_handler("RelationalExpression")
def _expr_relational(expr, var_to_col, engine):
    """Translate relational expressions (=, !=, <, >, <=, >=)."""
    op = expr.op
    if op not in _RELATIONAL_OPS:
        raise NotImplementedError(f"Relational op {op!r} not implemented")

    left_expr, right_expr = expr.expr, expr.other

    # Normalise: literal on left, variable on right -> swap
    left_type = _get_type_column(left_expr, var_to_col)
    right_type = _get_type_column(right_expr, var_to_col)
    if right_type is not None and left_type is None:
        left_expr, right_expr = right_expr, left_expr
        left_type, right_type = right_type, left_type
        op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(op, op)

    left = translate_expr(left_expr, var_to_col, engine)
    right = translate_expr(right_expr, var_to_col, engine)
    right_literal_type = _get_literal_type(right_expr)

    # Variable vs literal
    if left_type is not None and right_literal_type is not None:
        return _value_cmp_with_literal(left, left_type, right, right_literal_type, op)

    # Variable vs variable
    if left_type is not None and right_type is not None:
        return _value_cmp(left, left_type, right, right_type, op)

    # Simple comparison
    return _RELATIONAL_OPS[op](left, right)


def _get_type_column(expr, var_to_col):
    """Get the type column for an expression if it's a variable with type info."""
    if isinstance(expr, Variable):
        type_col_name = f"_ot_{expr}"
        return var_to_col.get(type_col_name)
    return None


def _get_literal_type(expr):
    """Get the datatype URI string for a Literal expression."""
    if isinstance(expr, Literal):
        if expr.datatype:
            return str(expr.datatype)
        if expr.language:
            return f"@{expr.language}"
        return str(XSD.string)
    return None


def _value_cmp(left, left_type, right, right_type, op):
    """Value-aware comparison between two variables."""
    numeric_types = [literal(t) for t in _NUMERIC_TYPES]
    both_numeric = and_(left_type.in_(numeric_types), right_type.in_(numeric_types))
    lexical_compatible = or_(
        and_(left_type.is_(None), right_type.is_(None)),
        and_(left_type.isnot(None), right_type.isnot(None), left_type == right_type),
    )

    left_real, right_real = func.cast(left, Float), func.cast(right, Float)
    cmp_op = _RELATIONAL_OPS[op]

    if op in ("=", "!="):
        equal_cond = or_(
            and_(both_numeric, left_real == right_real),
            and_(lexical_compatible, left == right),
        )
        return equal_cond if op == "=" else not_(equal_cond)

    return case(
        (both_numeric, cmp_op(left_real, right_real)), else_=cmp_op(left, right)
    )


def _value_cmp_with_literal(var_value, var_type, lit_value, lit_type_str, op):
    """Compare a variable to a literal with type awareness."""
    cmp_op = _RELATIONAL_OPS[op]

    if lit_type_str in _NUMERIC_TYPES:
        var_is_numeric = var_type.in_([literal(t) for t in _NUMERIC_TYPES])
        return and_(
            var_is_numeric,
            cmp_op(func.cast(var_value, Float), func.cast(lit_value, Float)),
        )

    return and_(var_type == literal(lit_type_str), cmp_op(var_value, lit_value))


def _translate_ebv_operand(operand_expr, var_to_col, engine):
    """Translate an operand applying EBV if it's a variable with type info."""
    ebv_cols = _needs_ebv(operand_expr, var_to_col)
    if ebv_cols:
        value_col, type_col = ebv_cols
        return _ebv(value_col, type_col)
    return translate_expr(operand_expr, var_to_col, engine)


@expr_handler("ConditionalAndExpression")
def _expr_and(expr, var_to_col, engine):
    """Translate && expressions with EBV semantics."""
    operands = [_translate_ebv_operand(expr.expr, var_to_col, engine)]
    operands.extend(_translate_ebv_operand(e, var_to_col, engine) for e in expr.other)
    return and_(*operands)


@expr_handler("ConditionalOrExpression")
def _expr_or(expr, var_to_col, engine):
    """Translate || expressions with EBV semantics."""
    operands = [_translate_ebv_operand(expr.expr, var_to_col, engine)]
    operands.extend(_translate_ebv_operand(e, var_to_col, engine) for e in expr.other)
    return or_(*operands)


@expr_handler("UnaryNot")
def _expr_not(expr, var_to_col, engine):
    """Translate ! expressions with EBV semantics."""
    return not_(_translate_ebv_operand(expr.expr, var_to_col, engine))


@expr_handler("AdditiveExpression")
@expr_handler("MultiplicativeExpression")
def _expr_binary_chain(expr, var_to_col, engine):
    """Translate chained binary expressions (+/-, *//)."""
    base = translate_expr(expr.expr, var_to_col, engine)
    for op, other in zip(getattr(expr, "op", []), getattr(expr, "other", [])):
        if op not in _BINARY_OPS:
            raise NotImplementedError(f"Binary op {op!r} not implemented")
        base = _BINARY_OPS[op](base, translate_expr(other, var_to_col, engine))
    return base


@expr_handler("InExpression")
def _expr_in(expr, var_to_col, engine):
    """Translate IN / NOT IN expressions."""
    lhs = translate_expr(expr.expr, var_to_col, engine)
    items = [translate_expr(v, var_to_col, engine) for v in expr.other]
    return lhs.not_in(items) if expr.notin else lhs.in_(items)


def _translate_builtin(expr, var_to_col, engine):
    """Translate SPARQL built-in functions (Builtin_XXX)."""
    fname = expr.name[8:].upper()

    raw_args = getattr(expr, "arg", None) or getattr(expr, "args", [])
    if not isinstance(raw_args, list):
        raw_args = [raw_args]
    args = [translate_expr(a, var_to_col, engine) for a in raw_args]

    if fname in _BUILTIN_SIMPLE:
        return _BUILTIN_SIMPLE[fname](*args)

    if fname in _BUILTIN_NOARG:
        return _BUILTIN_NOARG[fname]()

    if fname == "IF":
        return case((args[0], args[1]), else_=args[2])

    if fname == "BOUND":
        return args[0].isnot(None)

    if fname == "SUBSTR":
        string_arg = args[0] if args else translate_expr(expr.arg, var_to_col, engine)
        start = (
            translate_expr(expr.start, var_to_col, engine)
            if hasattr(expr, "start")
            else None
        )
        length = (
            translate_expr(expr.length, var_to_col, engine)
            if hasattr(expr, "length") and expr.length is not None
            else None
        )
        if length is not None:
            return func.substr(string_arg, start, length)
        if start is not None:
            return func.substr(string_arg, start)
        return func.substr(string_arg)

    if fname == "CONCAT":
        if not args:
            return literal("")
        result = args[0]
        for arg in args[1:]:
            result = result.concat(arg)
        return result

    if fname == "STR":
        return args[0]

    if fname == "STRSTARTS":
        s = translate_expr(expr.arg1, var_to_col, engine)
        prefix = translate_expr(expr.arg2, var_to_col, engine)
        return s.like(prefix.concat(literal("%")))

    if fname == "STRENDS":
        s = translate_expr(expr.arg1, var_to_col, engine)
        suffix = translate_expr(expr.arg2, var_to_col, engine)
        return s.like(literal("%").concat(suffix))

    if fname == "CONTAINS":
        s = translate_expr(expr.arg1, var_to_col, engine)
        frag = translate_expr(expr.arg2, var_to_col, engine)
        return s.like(literal("%").concat(frag).concat(literal("%")))

    if fname in ("REGEX", "REPLACE"):
        raise NotImplementedError(f"{fname} is dialect-specific and not implemented")

    if fname in _DATETIME_FIELDS:
        return func.extract(fname.lower(), *args)

    if fname in _HASH_FUNCS:
        return getattr(func, fname.lower())(*args)

    if fname == "SAMETERM":
        arg1 = translate_expr(expr.arg1, var_to_col, engine)
        arg2 = translate_expr(expr.arg2, var_to_col, engine)
        type1 = _get_type_column(expr.arg1, var_to_col)
        type2 = _get_type_column(expr.arg2, var_to_col)

        if type1 is not None and type2 is not None:
            return and_(
                arg1 == arg2,
                or_(and_(type1.is_(None), type2.is_(None)), type1 == type2),
            )
        return arg1 == arg2

    if fname in {"LANG", "DATATYPE", "ISIRI", "ISBLANK", "ISLITERAL", "ISNUMERIC"}:
        raise NotImplementedError(f"{fname} requires schema-specific mapping")

    raise NotImplementedError(f"SPARQL built-in function {fname} is not implemented")


# Register all Builtin_ handlers dynamically
for _builtin_name in [
    "Builtin_IF",
    "Builtin_BOUND",
    "Builtin_SUBSTR",
    "Builtin_CONCAT",
    "Builtin_STR",
    "Builtin_STRSTARTS",
    "Builtin_STRENDS",
    "Builtin_CONTAINS",
    "Builtin_REGEX",
    "Builtin_REPLACE",
    "Builtin_STRLEN",
    "Builtin_UCASE",
    "Builtin_LCASE",
    "Builtin_ABS",
    "Builtin_ROUND",
    "Builtin_CEIL",
    "Builtin_FLOOR",
    "Builtin_RAND",
    "Builtin_NOW",
    "Builtin_YEAR",
    "Builtin_MONTH",
    "Builtin_DAY",
    "Builtin_HOURS",
    "Builtin_MINUTES",
    "Builtin_SECONDS",
    "Builtin_MD5",
    "Builtin_SHA1",
    "Builtin_SHA256",
    "Builtin_SHA384",
    "Builtin_SHA512",
    "Builtin_COALESCE",
    "Builtin_sameTerm",
    "Builtin_LANG",
    "Builtin_DATATYPE",
    "Builtin_ISIRI",
    "Builtin_ISBLANK",
    "Builtin_ISLITERAL",
    "Builtin_ISNUMERIC",
]:
    _EXPRS[_builtin_name] = _translate_builtin


def _translate_aggregate(expr, var_to_col, engine):
    """Translate Aggregate_* expressions."""
    agg_var = getattr(expr, "vars", None)
    arg = translate_expr(agg_var, var_to_col, engine) if agg_var else literal(1)
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
        if engine is None:
            raise NotImplementedError("GROUP_CONCAT requires engine for dialect")
        f = dialect_funcs.get(engine.dialect.name)
        if not f:
            raise NotImplementedError(
                f"GROUP_CONCAT not supported for {engine.dialect.name}"
            )
        return f(arg.distinct(), sep) if distinct else f(arg, sep)

    raise NotImplementedError(f"Aggregate {expr.name!r} not implemented")


# Register aggregate handlers
for _agg_name in list(_AGG_FUNCS.keys()) + ["Aggregate_Group_Concat"]:
    _EXPRS[_agg_name] = _translate_aggregate


# =============================================================================
# Pattern Translation Registry
# =============================================================================

_PATTERNS: Dict[str, Callable[[CompValue, Context, Engine], QueryResult]] = {}


def pattern_handler(name: str):
    """Decorator to register a pattern handler."""

    def decorator(fn):
        _PATTERNS[name] = fn
        return fn

    return decorator


def translate_pattern(node: CompValue, ctx: Context, engine: Engine = None) -> QueryResult:
    """Dispatch to appropriate pattern handler."""
    if not hasattr(node, "name"):
        raise ValueError(f"Unknown pattern type: {node}")
    handler = _PATTERNS.get(node.name)
    if not handler:
        raise NotImplementedError(f"Pattern {node.name} not implemented")
    return handler(node, ctx, engine)


# =============================================================================
# Triple Pattern to Query
# =============================================================================


def _ot_condition(ot_column, term):
    """Generate object type condition for a concrete term."""
    o_type = term_to_object_type(term)
    return ot_column.is_(None) if o_type is None else ot_column == o_type


def triple_to_query(triple: tuple, ctx: Context) -> Select:
    """Convert a single triple pattern to a SELECT query."""
    s, p, o = triple
    columns, conditions = [], []
    graph_var = ctx.graph_term if isinstance(ctx.graph_term, Variable) else None

    # Subject
    if isinstance(s, Variable):
        if graph_var and s == graph_var:
            conditions.append(ctx.table.c.s == ctx.table.c.g)
        else:
            columns.append(ctx.table.c.s.label(str(s)))
    elif isinstance(s, BNode):
        columns.append(ctx.table.c.s.label(f"_bnode_{s}"))
    elif (s_str := term_to_string(s)) is not None:
        conditions.append(ctx.table.c.s == s_str)
    else:
        raise NotImplementedError(f"Subject type {type(s)} not implemented")

    # Predicate
    if isinstance(p, Variable):
        if p == s:
            conditions.append(ctx.table.c.p == ctx.table.c.s)
        elif graph_var and p == graph_var:
            conditions.append(ctx.table.c.p == ctx.table.c.g)
        else:
            columns.append(ctx.table.c.p.label(str(p)))
    elif isinstance(p, BNode):
        columns.append(ctx.table.c.p.label(f"_bnode_{p}"))
    elif (p_str := term_to_string(p)) is not None:
        conditions.append(ctx.table.c.p == p_str)
    else:
        raise NotImplementedError(f"Predicate type {type(p)} not implemented")

    # Object
    if isinstance(o, Variable):
        if o == s:
            conditions.append(ctx.table.c.o == ctx.table.c.s)
            conditions.append(ctx.table.c.ot.is_(None))
        elif o == p:
            conditions.append(ctx.table.c.o == ctx.table.c.p)
            conditions.append(ctx.table.c.ot.is_(None))
        elif graph_var and o == graph_var:
            conditions.append(ctx.table.c.o == ctx.table.c.g)
            conditions.append(ctx.table.c.ot.is_(None))
        else:
            columns.append(ctx.table.c.o.label(str(o)))
            columns.append(ctx.table.c.ot.label(f"_ot_{o}"))
    elif isinstance(o, BNode):
        columns.append(ctx.table.c.o.label(f"_bnode_{o}"))
    elif (o_str := term_to_string(o)) is not None:
        conditions.append(ctx.table.c.o == o_str)
        conditions.append(_ot_condition(ctx.table.c.ot, o))
    else:
        raise NotImplementedError(f"Object type {type(o)} not implemented")

    # Graph context
    if (g_filter := ctx.graph_filter) is not None:
        conditions.append(g_filter)
        if isinstance(ctx.graph_term, Variable):
            columns.append(ctx.table.c.g.label(str(ctx.graph_term)))

    # Ensure valid SELECT clause
    if not columns:
        columns = [literal(1).label("_exists_")]

    query = select(*columns).select_from(ctx.table)
    if conditions:
        query = query.where(*conditions)
    return query


def expand_triple(triple: tuple, ctx: Context) -> List[Select]:
    """Expand a triple pattern into one or more Select queries.

    Handles SequencePath and MulPath predicates.
    """
    s, p, o = triple

    if not isinstance(p, Path):
        return [triple_to_query(triple, ctx)]

    if isinstance(p, SequencePath):
        temp_vars = [Variable(f"_path_{id(p)}_{i}") for i in range(len(p.args) - 1)]
        expanded = zip([s] + temp_vars, p.args, temp_vars + [o])
        return [triple_to_query(t, ctx) for t in expanded]

    if isinstance(p, MulPath):
        return [_mulpath_to_cte(s, p, o, ctx)]

    raise NotImplementedError(f"Path type {type(p)} not implemented")


def _mulpath_to_cte(s, mulpath: MulPath, o, ctx: Context) -> Select:
    """Generate a recursive CTE for MulPath (transitive closure)."""
    pred_value = term_to_string(mulpath.path)
    if pred_value is None:
        raise NotImplementedError(
            f"MulPath with path type {type(mulpath.path)} not implemented"
        )

    s_value, o_value = term_to_string(s), term_to_string(o)
    o_type = term_to_object_type(o) if not isinstance(o, Variable) else None

    base_conditions = [ctx.table.c.p == pred_value]
    if (g_filter := ctx.graph_filter) is not None:
        base_conditions.append(g_filter)

    base_cols = [ctx.table.c.s, ctx.table.c.o, ctx.table.c.ot]
    if isinstance(ctx.graph_term, Variable):
        base_cols.append(ctx.table.c.g)

    base_query = select(*base_cols).where(and_(*base_conditions))

    ctx, cte_name = ctx.next_cte_name()
    path_cte = base_query.cte(name=f"path_{cte_name}", recursive=True)

    # Recursive conditions
    recursive_conditions = [ctx.table.c.p == pred_value]
    if ctx.graph_aware:
        if ctx.graph_term is None:
            recursive_conditions.append(ctx.table.c.g.is_(None))
        elif isinstance(ctx.graph_term, Variable):
            recursive_conditions.append(ctx.table.c.g == path_cte.c.g)
        else:
            recursive_conditions.append(ctx.table.c.g == str(ctx.graph_term))

    recursive_cols = [path_cte.c.s, ctx.table.c.o, ctx.table.c.ot]
    if isinstance(ctx.graph_term, Variable):
        recursive_cols.append(path_cte.c.g)

    recursive_query = (
        select(*recursive_cols)
        .select_from(path_cte.join(ctx.table, path_cte.c.o == ctx.table.c.s))
        .where(and_(*recursive_conditions))
    )

    path_cte = path_cte.union(recursive_query)

    # Build result columns
    result_columns = []
    if isinstance(s, Variable):
        result_columns.append(path_cte.c.s.label(str(s)))
    if isinstance(o, Variable):
        result_columns.append(path_cte.c.o.label(str(o)))
    if isinstance(ctx.graph_term, Variable):
        result_columns.append(path_cte.c.g.label(str(ctx.graph_term)))

    if not result_columns:
        result_columns = [path_cte.c.s, path_cte.c.o]

    final_query = select(*result_columns).select_from(path_cte)

    conditions = []
    if s_value is not None:
        conditions.append(path_cte.c.s == s_value)
    if o_value is not None:
        conditions.append(path_cte.c.o == o_value)
        if o_type is None:
            conditions.append(path_cte.c.ot.is_(None))
        else:
            conditions.append(path_cte.c.ot == o_type)

    if conditions:
        final_query = final_query.where(and_(*conditions))

    return final_query


# =============================================================================
# Join Utilities
# =============================================================================


def join_queries(queries: list, ctx: Context) -> QueryResult:
    """Natural join multiple queries on common variables (SPARQL semantics)."""
    if len(queries) == 1:
        return queries[0]

    ctes = [as_cte(q) for q in queries]

    def sparql_join_cond(left_col, right_col, is_internal):
        """SPARQL join: NULL matches anything for variables."""
        if is_internal:
            return left_col == right_col
        return or_(left_col.is_(None), right_col.is_(None), left_col == right_col)

    # Build join conditions for common columns
    conditions = [
        sparql_join_cond(left.c[col], right.c[col], col.startswith("_"))
        for left, right in itertools.combinations(ctes, 2)
        for col in set(left.c.keys()) & set(right.c.keys())
    ]

    # Collect all columns
    all_col_names = {name for cte in ctes for name in cte.c.keys()}
    cte_cols = {
        name: [cte.c[name] for cte in ctes if name in cte.c.keys()]
        for name in all_col_names
    }

    def project_col(name, col_list):
        if len(col_list) > 1 and not name.startswith("_"):
            return func.coalesce(*col_list).label(name)
        return col_list[0]

    final_cols = [project_col(name, col_list) for name, col_list in cte_cols.items()]

    return select(*final_cols).select_from(*ctes).where(*conditions)


# =============================================================================
# Pattern Handlers
# =============================================================================


@pattern_handler("BGP")
def _pattern_bgp(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Basic Graph Pattern."""
    triples = node["triples"]

    if not triples:
        # Empty BGP handling
        if ctx.graph_term is None:
            if not ctx.graph_aware:
                return select(literal(1).label("_exists_"))
            return (
                select(literal(1).label("_exists_"))
                .where(ctx.table.c.g.is_(None))
                .limit(1)
            )
        if isinstance(ctx.graph_term, Variable):
            return (
                select(ctx.table.c.g.label(str(ctx.graph_term)))
                .where(ctx.table.c.g.isnot(None))
                .distinct()
            )
        return (
            select(literal(1).label("_exists_"))
            .where(ctx.table.c.g == str(ctx.graph_term))
            .limit(1)
        )

    queries = [q for triple in triples for q in expand_triple(triple, ctx)]
    return join_queries(queries, ctx)


@pattern_handler("Graph")
def _pattern_graph(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate GRAPH pattern - simply updates context."""
    if not ctx.graph_aware:
        raise ValueError("Graph patterns require graph_aware=True")
    return translate_pattern(node["p"], ctx.with_graph(node["term"]), engine)


@pattern_handler("Project")
def _pattern_project(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Project (SELECT) operation."""
    project_vars = node["PV"]
    base_query = translate_pattern(node["p"], ctx, engine)
    base_cols = col_names(base_query)
    project_var_names = {str(var) for var in project_vars}

    if base_cols == project_var_names:
        return base_query

    if not project_vars:
        if isinstance(base_query, CTE):
            return select(literal(1).label("__placeholder__")).select_from(base_query)
        return base_query.with_only_columns(literal(1).label("__placeholder__"))

    if isinstance(base_query, CTE):
        var_columns = [
            base_query.c[str(var)] if str(var) in base_cols else null().label(str(var))
            for var in project_vars
        ]
        return select(*var_columns).select_from(base_query)

    var_to_col = {col.key: col for col in base_query.selected_columns}
    var_columns = [
        var_to_col[str(var)] if str(var) in var_to_col else null().label(str(var))
        for var in project_vars
    ]
    return base_query.with_only_columns(*var_columns)


@pattern_handler("Filter")
def _pattern_filter(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Filter operation."""
    inner_query = translate_pattern(node["p"], ctx, engine)

    if isinstance(inner_query, CTE):
        var_to_col = cols(inner_query)
        base_select = select(*inner_query.c).select_from(inner_query)
    else:
        var_to_col = cols(inner_query)
        base_select = inner_query

    # Apply EBV if the filter expression is a bare variable
    filter_expr = node["expr"]
    ebv_cols = _needs_ebv(filter_expr, var_to_col)
    if ebv_cols:
        value_col, type_col = ebv_cols
        sql_condition = _ebv(value_col, type_col)
    else:
        sql_condition = translate_expr(filter_expr, var_to_col, engine)
    return base_select.where(sql_condition)


@pattern_handler("Distinct")
def _pattern_distinct(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Distinct operation."""
    inner_query = translate_pattern(node["p"], ctx, engine)

    if isinstance(inner_query, CTE):
        return select(*inner_query.c).select_from(inner_query).distinct()
    return inner_query.distinct()


@pattern_handler("Extend")
def _pattern_extend(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Extend (BIND) operation."""
    inner_query = translate_pattern(node["p"], ctx, engine)
    cte = as_cte(inner_query)
    var_to_col = cols(cte)

    sql_expr = translate_expr(node["expr"], var_to_col, engine)
    result_columns = list(cte.c) + [sql_expr.label(str(node["var"]))]

    return select(*result_columns).select_from(cte)


@pattern_handler("OrderBy")
def _pattern_order_by(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate OrderBy operation."""
    inner_query = translate_pattern(node["p"], ctx, engine)

    if isinstance(inner_query, CTE):
        var_to_col = cols(inner_query)
        base_select = select(*inner_query.c).select_from(inner_query)
    else:
        var_to_col = cols(inner_query)
        base_select = inner_query

    order_clauses = []
    for cond in node["expr"]:
        if isinstance(cond, CompValue) and cond.name == "OrderCondition":
            sql_expr = translate_expr(cond.expr, var_to_col, engine)
            order_clauses.append(
                desc(sql_expr) if cond.order == "DESC" else asc(sql_expr)
            )
        else:
            order_clauses.append(asc(translate_expr(cond, var_to_col, engine)))

    return base_select.order_by(*order_clauses)


@pattern_handler("Join")
def _pattern_join(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Join operation (natural join)."""
    left = translate_pattern(node["p1"], ctx, engine)
    right = translate_pattern(node["p2"], ctx, engine)
    return join_queries([left, right], ctx)


@pattern_handler("Slice")
def _pattern_slice(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Slice (LIMIT/OFFSET) operation."""
    start = getattr(node, "start", 0)
    length = getattr(node, "length", None)
    inner_query = translate_pattern(node["p"], ctx, engine)

    if isinstance(inner_query, CTE):
        result = select(*inner_query.c).select_from(inner_query)
    else:
        result = inner_query

    if length is not None:
        result = result.limit(length)
    if start > 0:
        result = result.offset(start)

    return result


@pattern_handler("LeftJoin")
def _pattern_left_join(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate LeftJoin (OPTIONAL) operation."""
    left_query = translate_pattern(node["p1"], ctx, engine)
    right_query = translate_pattern(node["p2"], ctx, engine)
    filter_expr = node.get("expr")

    left_cte = as_cte(left_query)
    right_cte = as_cte(right_query)

    var_to_col = {**cols(right_cte), **cols(left_cte)}
    common_cols = set(left_cte.c.keys()) & set(right_cte.c.keys())

    join_conditions = [left_cte.c[col] == right_cte.c[col] for col in common_cols]

    if filter_expr is not None:
        if not (
            isinstance(filter_expr, CompValue) and filter_expr.name == "TrueFilter"
        ):
            join_conditions.append(translate_expr(filter_expr, var_to_col, engine))

    all_columns = list(left_cte.c) + [
        col for name, col in right_cte.c.items() if name not in common_cols
    ]

    on_condition = and_(*join_conditions) if join_conditions else true()

    return select(*all_columns).select_from(left_cte.outerjoin(right_cte, on_condition))


@pattern_handler("Union")
def _pattern_union(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Union operation (SPARQL UNION -> SQL UNION ALL)."""
    left_q = translate_pattern(node["p1"], ctx, engine)
    right_q = translate_pattern(node["p2"], ctx, engine)
    left_cte = as_cte(left_q)
    right_cte = as_cte(right_q)

    all_cols = sorted(col_names(left_q) | col_names(right_q))

    def padded_select(cte):
        return select(
            *[
                cte.c[c].label(c) if c in cte.c.keys() else null().label(c)
                for c in all_cols
            ]
        ).select_from(cte)

    return padded_select(left_cte).union_all(padded_select(right_cte)).cte()


@pattern_handler("Group")
def _pattern_group(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Group operation."""
    return translate_pattern(node["p"], ctx, engine)


@pattern_handler("AggregateJoin")
def _pattern_aggregate_join(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate AggregateJoin (GROUP BY with aggregates)."""
    inner_query = translate_pattern(node["p"], ctx, engine)
    cte = as_cte(inner_query)
    var_to_col = cols(cte)

    group_vars = getattr(node["p"], "expr", None) or []
    group_cols = [cte.c[str(v)] for v in group_vars if str(v) in var_to_col]

    agg_cols = [
        translate_expr(agg, var_to_col, engine).label(str(agg.res)) for agg in node["A"]
    ]

    result = select(*agg_cols).select_from(cte)
    return result.group_by(*group_cols) if group_cols else result


# =============================================================================
# Schema Helper
# =============================================================================


def create_triples_table(
    metadata: MetaData, table_name: str, graph_aware: bool = False
) -> Table:
    """Create the triples table definition."""
    columns = [
        Column("s", String, nullable=False, primary_key=True),
        Column("p", String, nullable=False, primary_key=True),
        Column("o", String, nullable=False, primary_key=True),
        Column("ot", String, nullable=True, primary_key=True),
    ]
    if graph_aware:
        columns.append(Column("g", String, nullable=True, primary_key=True))

    return Table(quoted_name(table_name, quote=False), metadata, *columns)


# =============================================================================
# Main Translator API
# =============================================================================


class Translator:
    """Translate and execute SPARQL queries against a SQL database.

    The table name specified is treated as a SPARQL Dataset containing
    a default graph and optionally named graphs (if graph_aware=True).
    """

    def __init__(
        self,
        engine: Engine,
        table_name: str = "triples",
        create_table: bool = False,
        graph_aware: bool = False,
    ):
        self.engine = engine
        self.metadata = MetaData()
        self.graph_aware = graph_aware
        self.table = create_triples_table(self.metadata, table_name, graph_aware)

        if create_table:
            self.metadata.create_all(self.engine)

    def _ctx(self) -> Context:
        """Create a fresh translation context."""
        return Context(table=self.table, graph_aware=self.graph_aware)

    def execute(self, sparql_query: str) -> Union[CursorResult, bool]:
        """Translate and execute a SPARQL query.

        Returns:
            CursorResult for SELECT queries
            bool for ASK queries
        """
        query_type, sql_query = self.translate(sparql_query)

        with self.engine.connect() as conn:
            if query_type == "SelectQuery":
                return conn.execute(sql_query)
            if query_type == "AskQuery":
                return conn.execute(sql_query).scalar()

        raise NotImplementedError(f"Query type {query_type} not implemented")

    def translate(self, sparql_string: str) -> tuple[str, Select]:
        """Translate a SPARQL query string to a SQLAlchemy query.

        Returns:
            Tuple of (query_type, sql_query)
        """
        query_tree = parser.parseQuery(sparql_string)
        query_algebra = algebra.translateQuery(query_tree).algebra
        query_type = query_algebra.name

        if query_type == "SelectQuery":
            return query_type, self._translate_select(query_algebra)
        if query_type == "AskQuery":
            return query_type, self._translate_ask(query_algebra)

        raise NotImplementedError(f"Query type {query_type} not implemented")

    def _translate_select(self, select_query: CompValue) -> Select:
        """Translate a SelectQuery."""
        base_query = translate_pattern(select_query["p"], self._ctx(), self.engine)

        if isinstance(base_query, CTE):
            return select(*base_query.c).select_from(base_query)
        return base_query

    def _translate_ask(self, ask_query: CompValue) -> Select:
        """Translate an AskQuery to SELECT EXISTS(...)."""
        base_query = translate_pattern(ask_query["p"], self._ctx(), self.engine)

        if isinstance(base_query, CTE):
            subq = select(literal(1)).select_from(base_query).limit(1)
        else:
            subq = base_query.limit(1)

        return select(exists(subq).label("result"))


# =============================================================================
# Convenience Functions
# =============================================================================


def create_databricks_engine(
    server_hostname: str, http_path: str, access_token: str, **engine_kwargs
) -> Engine:
    """Create a SQLAlchemy engine for Databricks."""
    engine_uri = (
        f"databricks://token:{access_token}@{server_hostname}?http_path={http_path}"
    )
    return create_engine(engine_uri, **engine_kwargs)
