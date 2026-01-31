"""SPARQL to SQL translation using SQLAlchemy.

This module translates SPARQL algebra (from rdflib) to SQL queries.
Design principles:
- Explicit context passing (no mutable instance state)
- Unified intermediate representation (always use CTEs)
- Registry-based dispatch (declarative handler registration)
- Separation of concerns (schema, translation, execution are distinct)
"""

import itertools
import re
from dataclasses import dataclass, replace, field
from typing import Callable, Dict, List, Optional, Union

from rdflib import BNode, Literal, RDF, URIRef, XSD
from rdflib.paths import (
    AlternativePath,
    InvPath,
    MulPath,
    NegatedPath,
    Path,
    SequencePath,
)
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
    union,
)
from sqlalchemy.sql.selectable import SelectBase


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
            return f"@{term.language.lower()}"  # Language tags are case-insensitive (BCP 47)
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
    # When True, graph variable is not projected inside GRAPH pattern (proper SPARQL scoping)
    _inside_graph_var_pattern: bool = False
    _cte_count: int = field(default=0, compare=False)

    def with_graph(self, term, inside_var_pattern: bool = False) -> "Context":
        """Return new context with graph term set."""
        return replace(
            self, graph_term=term, _inside_graph_var_pattern=inside_var_pattern
        )

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

# Marker column for empty SPARQL projections (SQL requires at least one column)
EMPTY_PROJECTION_MARKER = "__empty__"

# Internal column for graph variable scoping (renamed to user variable in _pattern_graph)
INTERNAL_GRAPH_COLUMN = "__graph__"

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


def _project_columns(
    col_map: Dict[str, Column], var_names: list[str], include_ot: bool = False
) -> list[Column]:
    """Build column list for projection, with optional _ot_* type columns.

    Args:
        col_map: Column name -> Column mapping (from cols())
        var_names: Variable names to project
        include_ot: If True, include _ot_* type columns for term identity semantics

    Returns:
        List of columns (with NULL for missing vars)
    """
    result = []
    for var in var_names:
        col = col_map.get(var)
        result.append(col if col is not None else null().label(var))
        if include_ot:
            ot_col = col_map.get(f"_ot_{var}")
            if ot_col is not None:
                result.append(ot_col)
    return result


def _apply_distinct(query: QueryResult, columns: list[Column] = None) -> QueryResult:
    """Apply DISTINCT to a query, optionally projecting specific columns.

    Args:
        query: Source query (CTE or Select)
        columns: If provided, project these columns; otherwise use all columns

    Returns:
        Query with DISTINCT applied
    """
    if columns is None:
        if isinstance(query, CTE):
            return select(*query.c).select_from(query).distinct()
        return query.distinct()

    if isinstance(query, CTE):
        return select(*columns).select_from(query).distinct()
    return query.with_only_columns(*columns).distinct()


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

# Supported SPARQL REGEX flags (from XPath/XQuery spec)
# i = case insensitive, m = multiline, s = dot matches newlines, x = extended
# q = quote (treat pattern as literal string, escaping all special chars)
_REGEX_FLAGS = {"i", "m", "s", "x", "q"}


def _translate_regex(expr, var_to_col, engine):
    """Translate SPARQL REGEX function to dialect-specific SQL.

    SPARQL semantics: REGEX operates on simple literals (strings).
    Applying REGEX to a URI or blank node is a type error and should
    evaluate to false (filtering out the row).

    Supported dialects:
    - PostgreSQL: Uses ~ (case-sensitive) or ~* (case-insensitive) operators
    - Databricks: Uses rlike operator with (?flags) prefix in pattern
    - SQLite: Uses regexp function with (?flags) prefix in pattern
    """
    text = translate_expr(expr.text, var_to_col, engine)
    flags = str(expr.flags) if hasattr(expr, "flags") and expr.flags else ""

    # Validate flags
    invalid_flags = set(flags) - _REGEX_FLAGS
    if invalid_flags:
        raise NotImplementedError(f"Unsupported REGEX flags: {invalid_flags}")

    # Handle 'q' flag (quote) - escape all regex metacharacters in the pattern
    # This makes the pattern match literally without any regex interpretation
    if "q" in flags:
        if not isinstance(expr.pattern, Literal):
            raise NotImplementedError(
                "REGEX 'q' flag only supported with literal patterns"
            )
        escaped_pattern = re.escape(str(expr.pattern))
        pattern = literal(escaped_pattern)
        flags = flags.replace("q", "")  # Remove q flag, handled via escaping
    else:
        pattern = translate_expr(expr.pattern, var_to_col, engine)

    if engine is None:
        raise NotImplementedError("REGEX requires engine for dialect detection")

    dialect = engine.dialect.name

    if dialect == "postgresql":
        regex_cond = _regex_postgresql(text, pattern, flags)
    elif dialect == "databricks":
        regex_cond = _regex_databricks(text, pattern, flags)
    elif dialect == "sqlite":
        regex_cond = _regex_sqlite(text, pattern, flags)
    else:
        raise NotImplementedError(f"REGEX not supported for dialect: {dialect}")

    # If the text argument is a variable, ensure we only match literals (not URIs)
    # URIs have NULL in the type column, literals have a type value
    type_col = _get_type_column(expr.text, var_to_col)
    if type_col is not None:
        # Only match if the value is a literal (type column is not null)
        return and_(type_col.isnot(None), regex_cond)

    return regex_cond


def _regex_postgresql(text, pattern, flags):
    """Generate PostgreSQL regex expression.

    PostgreSQL uses POSIX regex operators:
    - ~ for case-sensitive match
    - ~* for case-insensitive match
    - Other flags are embedded in the pattern using (?flags) syntax
    """
    # For PostgreSQL, case-insensitivity uses ~* operator
    # Other flags (m, s, x) are embedded in the pattern
    case_insensitive = "i" in flags
    other_flags = flags.replace("i", "")

    if other_flags:
        # Embed remaining flags in pattern: (?ms)pattern
        pattern = literal(f"(?{other_flags})").concat(pattern)

    if case_insensitive:
        return text.op("~*")(pattern)
    else:
        return text.op("~")(pattern)


def _regex_databricks(text, pattern, flags):
    """Generate Databricks regex expression.

    Databricks uses rlike operator with Java regex syntax.
    All flags are embedded in the pattern using (?flags) syntax.
    """
    if flags:
        # Embed all flags in pattern: (?ims)pattern
        pattern = literal(f"(?{flags})").concat(pattern)

    return text.op("rlike")(pattern)


def _regex_sqlite(text, pattern, flags):
    """Generate SQLite regex expression.

    SQLite uses the regexp function (must be registered on the connection).
    Flags are embedded in the pattern using (?flags) syntax.

    Note: The regexp function must be registered using connection.create_function().
    Python's sqlite3 module can register Python's re.search for this purpose.
    """
    if flags:
        # Embed all flags in pattern: (?ims)pattern
        pattern = literal(f"(?{flags})").concat(pattern)

    return func.regexp(pattern, text)


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

# Pre-computed SQL literals for type comparisons (avoid repeated list comprehensions)
_NUMERIC_TYPE_LITERALS = tuple(literal(t) for t in _NUMERIC_TYPES)
_ORDERABLE_TYPES = {str(XSD.string), str(XSD.dateTime), str(XSD.date), str(XSD.time)}
_ORDERABLE_TYPE_LITERALS = tuple(
    literal(str(t)) for t in (XSD.string, XSD.dateTime, XSD.date, XSD.time)
)

# =============================================================================
# Boolean Expression Detection (for projection)
# =============================================================================

# Expressions that return boolean values (must be converted to "true"/"false" when projected)
_BOOLEAN_EXPRS = {
    "RelationalExpression",
    "ConditionalAndExpression",
    "ConditionalOrExpression",
    "UnaryNot",
    "Builtin_BOUND",
    "Builtin_SAMETERM",
    "Builtin_isIRI",
    "Builtin_isURI",
    "Builtin_isBLANK",
    "Builtin_isLITERAL",
    "Builtin_isNUMERIC",
    "Builtin_REGEX",
    "Builtin_CONTAINS",
    "Builtin_STRSTARTS",
    "Builtin_STRENDS",
}


def _is_boolean_expr(expr) -> bool:
    """Check if an expression returns a boolean value."""
    if isinstance(expr, Literal):
        return expr.datatype == XSD.boolean
    return isinstance(expr, CompValue) and expr.name in _BOOLEAN_EXPRS


def _bool_to_xsd_string(sql_expr):
    """Wrap a boolean SQL expression to return XSD boolean string values."""
    return case((sql_expr, literal("true")), else_=literal("false"))


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
    right_literal_type = term_to_object_type(right_expr)

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


# Boolean value semantics: "true"/"1" are equivalent, "false"/"0" are equivalent
_BOOL_WELL_FORMED = tuple(literal(v) for v in ("true", "false", "1", "0"))
_bool_is_true = lambda v: or_(v == literal("true"), v == literal("1"))


def _bool_eq(left, right):
    """Boolean equality: normalised comparison if well-formed, else same-term."""
    both_well_formed = and_(left.in_(_BOOL_WELL_FORMED), right.in_(_BOOL_WELL_FORMED))
    return case(
        (both_well_formed, _bool_is_true(left) == _bool_is_true(right)),
        else_=(left == right),
    )


def _value_cmp(left, left_type, right, right_type, op):
    """Value-aware comparison between two variables.

    Uses CASE for efficient SQL with short-circuit evaluation. Comparability rules:
    - URI/bnodes (type IS NULL): compare lexically with each other
    - Numeric types: compare by value (with ill-formed literal guard)
    - Boolean: normalise (true/1 and false/0 are equivalent); ill-formed uses same-term
    - Same orderable type (string, dateTime, date, time): compare lexically
    - Same language tag: compare lexically
    - Different term kinds (URI vs literal, lang vs typed): always != (type error for ordering)
    - Unknown types: same-term equality only; type error otherwise
    """
    cmp_op = _RELATIONAL_OPS[op]
    left_v, right_v = func.cast(left, Float), func.cast(right, Float)

    # Comparability predicates (ordered for CASE short-circuit efficiency)
    both_uri = and_(left_type.is_(None), right_type.is_(None))
    both_numeric = and_(
        left_type.in_(_NUMERIC_TYPE_LITERALS), right_type.in_(_NUMERIC_TYPE_LITERALS)
    )
    both_boolean = and_(
        left_type == literal(str(XSD.boolean)), right_type == literal(str(XSD.boolean))
    )
    same_orderable = and_(
        left_type.in_(_ORDERABLE_TYPE_LITERALS), left_type == right_type
    )
    same_lang = and_(left_type.like("@%"), left_type == right_type)

    if op == "=":
        # Guard for ill-formed numerics: both cast to 0 but different lexical form
        numeric_eq = and_(left_v == right_v, or_(left_v != literal(0), left == right))
        return case(
            (both_uri, left == right),
            (both_numeric, numeric_eq),
            (both_boolean, _bool_eq(left, right)),
            (or_(same_orderable, same_lang), left == right),
            (left_type == right_type, left == right),  # same-term for unknown types
            else_=literal(False),
        )

    if op == "!=":
        # Different term kinds are definitively unequal
        l_uri, r_uri = left_type.is_(None), right_type.is_(None)
        l_lang, r_lang = left_type.like("@%"), right_type.like("@%")
        different_kinds = or_(
            and_(l_uri, not_(r_uri)),
            and_(not_(l_uri), r_uri),
            and_(l_lang, not_(or_(r_uri, r_lang))),
            and_(not_(or_(l_uri, l_lang)), r_lang),
        )
        return case(
            (different_kinds, literal(True)),
            (both_numeric, left_v != right_v),
            (both_boolean, not_(_bool_eq(left, right))),
            (both_uri, left != right),
            (or_(same_orderable, same_lang), left != right),
            else_=literal(False),
        )

    # Ordering (<, >, <=, >=): numeric by value, orderable by lexical, else type error
    return case(
        (both_numeric, cmp_op(left_v, right_v)),
        (same_orderable, cmp_op(left, right)),
        else_=null(),
    )


def _value_cmp_with_literal(var_value, var_type, lit_value, lit_type_str, op):
    """Compare a variable to a literal with type awareness.

    Since the literal type is known at translation time, we generate optimised SQL
    without runtime type dispatch. Comparability rules:
    - Numeric: compare by value if variable is also numeric
    - Boolean: normalise both values (true/1 and false/0 are equivalent)
    - Orderable (string, dateTime, date, time) or lang-tagged: compare lexically
    - Unknown type: same-term equality only; type error otherwise
    """
    cmp_op = _RELATIONAL_OPS[op]

    if lit_type_str in _NUMERIC_TYPES:
        return and_(
            var_type.in_(_NUMERIC_TYPE_LITERALS),
            cmp_op(func.cast(var_value, Float), func.cast(lit_value, Float)),
        )

    if lit_type_str == str(XSD.boolean):
        is_bool = var_type == literal(lit_type_str)
        eq = _bool_eq(var_value, literal(lit_value))
        return and_(is_bool, eq if op == "=" else not_(eq))

    if lit_type_str in _ORDERABLE_TYPES or lit_type_str.startswith("@"):
        return and_(var_type == literal(lit_type_str), cmp_op(var_value, lit_value))

    # Unknown type: only same-term equality is valid
    if op == "=":
        return and_(var_type == literal(lit_type_str), var_value == lit_value)
    return literal(False)


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


@expr_handler("UnaryPlus")
def _expr_unary_plus(expr, var_to_col, engine):
    """Translate + prefix (identity for numeric values)."""
    return translate_expr(expr.expr, var_to_col, engine)


@expr_handler("UnaryMinus")
def _expr_unary_minus(expr, var_to_col, engine):
    """Translate - prefix (negation for numeric values)."""
    return -translate_expr(expr.expr, var_to_col, engine)


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


def _type_test_variable(fname, type_col, value_col):
    """Generate SQL condition for type test (isLiteral, isBlank, etc.) on a variable."""
    is_blank = value_col.like(literal("_:%"))
    numeric_types = [literal(t) for t in _NUMERIC_TYPES]

    if fname == "ISLITERAL":
        return type_col.isnot(None) if type_col is not None else not_(true())
    if fname == "ISBLANK":
        return and_(type_col.is_(None), is_blank) if type_col is not None else is_blank
    if fname in {"ISIRI", "ISURI"}:
        not_blank = not_(is_blank)
        return (
            and_(type_col.is_(None), not_blank) if type_col is not None else not_blank
        )
    if fname == "ISNUMERIC":
        return type_col.in_(numeric_types) if type_col is not None else not_(true())
    return not_(true())


def _type_test_constant(fname, arg):
    """Evaluate type test on a constant RDF term. Returns bool."""
    if fname == "ISLITERAL":
        return isinstance(arg, Literal)
    if fname == "ISBLANK":
        return isinstance(arg, BNode)
    if fname in {"ISIRI", "ISURI"}:
        return isinstance(arg, URIRef)
    if fname == "ISNUMERIC":
        return isinstance(arg, Literal) and str(arg.datatype or "") in _NUMERIC_TYPES
    return False


# =============================================================================
# Built-in Function Handler Registry
# =============================================================================

_BUILTINS: Dict[str, Callable] = {}


def builtin_handler(*names: str):
    """Decorator to register a built-in function handler.

    Handlers receive (expr, raw_args, args, var_to_col, engine) where:
      - expr: the original CompValue expression
      - raw_args: unevaluated argument expressions
      - args: translated SQLAlchemy column expressions
      - var_to_col: variable-to-column mapping
      - engine: SQLAlchemy engine (may be None)
    """

    def decorator(fn):
        for name in names:
            _BUILTINS[name] = fn
        return fn

    return decorator


def _translate_builtin(expr, var_to_col, engine):
    """Translate SPARQL built-in functions (Builtin_XXX)."""
    fname = expr.name[8:].upper()

    raw_args = getattr(expr, "arg", None) or getattr(expr, "args", [])
    if not isinstance(raw_args, list):
        raw_args = [raw_args]
    args = [translate_expr(a, var_to_col, engine) for a in raw_args]

    # Dispatch to registered handler
    handler = _BUILTINS.get(fname)
    if handler:
        return handler(expr, raw_args, args, var_to_col, engine)

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
    "Builtin_LANGMATCHES",
    "Builtin_DATATYPE",
    "Builtin_isIRI",
    "Builtin_isURI",
    "Builtin_isBLANK",
    "Builtin_isLITERAL",
    "Builtin_isNUMERIC",
]:
    _EXPRS[_builtin_name] = _translate_builtin


# --- Built-in Function Handlers ---


@builtin_handler("STRLEN", "UCASE", "LCASE", "ABS", "ROUND", "CEIL", "FLOOR")
def _builtin_simple(expr, raw_args, args, var_to_col, engine):
    """Handle simple built-ins that map directly to SQL functions."""
    fname = expr.name[8:].upper()
    return _BUILTIN_SIMPLE[fname](*args)


@builtin_handler("COALESCE")
def _builtin_coalesce(expr, raw_args, args, var_to_col, engine):
    """Handle COALESCE (SQLite requires at least 2 arguments)."""
    return func.coalesce(*args, null()) if len(args) < 2 else func.coalesce(*args)


@builtin_handler("MD5", "SHA1", "SHA256", "SHA384", "SHA512")
def _builtin_hash(expr, raw_args, args, var_to_col, engine):
    """Handle hash functions."""
    fname = expr.name[8:].upper()
    return getattr(func, fname.lower())(*args)


@builtin_handler("RAND", "NOW")
def _builtin_noarg(expr, raw_args, args, var_to_col, engine):
    """Handle no-argument built-ins."""
    fname = expr.name[8:].upper()
    return _BUILTIN_NOARG[fname]()


@builtin_handler("YEAR", "MONTH", "DAY", "HOURS", "MINUTES", "SECONDS")
def _builtin_datetime(expr, raw_args, args, var_to_col, engine):
    """Handle datetime field extraction."""
    fname = expr.name[8:].upper()
    return func.extract(fname.lower(), *args)


@builtin_handler("IF")
def _builtin_if(expr, raw_args, args, var_to_col, engine):
    """Handle IF(condition, then, else)."""
    return case((args[0], args[1]), else_=args[2])


@builtin_handler("BOUND")
def _builtin_bound(expr, raw_args, args, var_to_col, engine):
    """Handle BOUND(?var) - true if variable is bound."""
    return args[0].isnot(None)


@builtin_handler("SUBSTR")
def _builtin_substr(expr, raw_args, args, var_to_col, engine):
    """Handle SUBSTR(string, start [, length])."""
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
    return (
        func.substr(string_arg, start) if start is not None else func.substr(string_arg)
    )


@builtin_handler("CONCAT")
def _builtin_concat(expr, raw_args, args, var_to_col, engine):
    """Handle CONCAT(str1, str2, ...)."""
    if not args:
        return literal("")
    result = args[0]
    for arg in args[1:]:
        result = result.concat(arg)
    return result


@builtin_handler("STR")
def _builtin_str(expr, raw_args, args, var_to_col, engine):
    """Handle STR(term) - convert to string representation."""
    return args[0]


@builtin_handler("STRSTARTS")
def _builtin_strstarts(expr, raw_args, args, var_to_col, engine):
    """Handle STRSTARTS(string, prefix)."""
    s = translate_expr(expr.arg1, var_to_col, engine)
    prefix = translate_expr(expr.arg2, var_to_col, engine)
    return s.like(prefix.concat(literal("%")))


@builtin_handler("STRENDS")
def _builtin_strends(expr, raw_args, args, var_to_col, engine):
    """Handle STRENDS(string, suffix)."""
    s = translate_expr(expr.arg1, var_to_col, engine)
    suffix = translate_expr(expr.arg2, var_to_col, engine)
    return s.like(literal("%").concat(suffix))


@builtin_handler("CONTAINS")
def _builtin_contains(expr, raw_args, args, var_to_col, engine):
    """Handle CONTAINS(string, fragment)."""
    s = translate_expr(expr.arg1, var_to_col, engine)
    frag = translate_expr(expr.arg2, var_to_col, engine)
    return s.like(literal("%").concat(frag).concat(literal("%")))


@builtin_handler("REGEX")
def _builtin_regex(expr, raw_args, args, var_to_col, engine):
    """Handle REGEX(string, pattern [, flags])."""
    return _translate_regex(expr, var_to_col, engine)


@builtin_handler("REPLACE")
def _builtin_replace(expr, raw_args, args, var_to_col, engine):
    """Handle REPLACE - not yet implemented."""
    raise NotImplementedError("REPLACE is dialect-specific and not implemented")


@builtin_handler("SAMETERM")
def _builtin_sameterm(expr, raw_args, args, var_to_col, engine):
    """Handle sameTerm(a, b) - strict term equality."""
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


@builtin_handler("ISLITERAL", "ISBLANK", "ISIRI", "ISURI", "ISNUMERIC")
def _builtin_type_test(expr, raw_args, args, var_to_col, engine):
    """Handle type test functions (isLiteral, isBlank, isIRI, isURI, isNumeric)."""
    fname = expr.name[8:].upper()
    arg_expr = raw_args[0] if raw_args else None
    if isinstance(arg_expr, Variable):
        type_col = _get_type_column(arg_expr, var_to_col)
        value_col = args[0]
        return _type_test_variable(fname, type_col, value_col)
    return true() if _type_test_constant(fname, arg_expr) else not_(true())


@builtin_handler("LANG")
def _builtin_lang(expr, raw_args, args, var_to_col, engine):
    """Handle LANG(literal) - returns language tag or empty string."""
    # LANG returns the language tag of a literal, or "" if none
    # For non-literals (URIs, blank nodes), return NULL to trigger error semantics
    arg_expr = raw_args[0] if raw_args else None
    type_col = _get_type_column(arg_expr, var_to_col)
    if type_col is not None:
        # ot IS NULL means URI/blank node -> NULL (error)
        # ot LIKE '@%' means language-tagged -> extract tag
        # otherwise (typed literal) -> empty string
        return case(
            (type_col.is_(None), null()),
            (type_col.like("@%"), func.substr(type_col, 2)),
            else_=literal(""),
        )
    # Constant: check if it's a language-tagged literal
    if isinstance(arg_expr, Literal) and arg_expr.language:
        return literal(arg_expr.language.lower())
    if isinstance(arg_expr, Literal):
        return literal("")
    # Non-literal constant -> NULL (error)
    return null()


@builtin_handler("LANGMATCHES")
def _builtin_langmatches(expr, raw_args, args, var_to_col, engine):
    """Handle LANGMATCHES(lang_tag, lang_range) - BCP 47 language matching."""
    lang_tag = translate_expr(expr.arg1, var_to_col, engine)
    range_val = translate_expr(expr.arg2, var_to_col, engine)

    # Handle "*" wildcard: matches any non-empty language tag
    if isinstance(expr.arg2, Literal) and str(expr.arg2) == "*":
        return lang_tag != literal("")

    # BCP 47 basic filtering: exact match OR prefix-with-hyphen match
    # e.g., "en" matches "en" and "en-gb", but "en-gb" doesn't match "en"
    lang_lower = func.lower(lang_tag)
    range_lower = func.lower(range_val)
    return or_(
        lang_lower == range_lower,
        lang_lower.like(range_lower.concat(literal("-%"))),
    )


@builtin_handler("DATATYPE")
def _builtin_datatype(expr, raw_args, args, var_to_col, engine):
    """Handle DATATYPE(literal) - returns datatype IRI."""
    # DATATYPE returns the datatype IRI of a literal
    # For non-literals (URIs, blank nodes), return NULL (error semantics)
    arg_expr = raw_args[0] if raw_args else None
    type_col = _get_type_column(arg_expr, var_to_col)
    if type_col is not None:
        # ot IS NULL means URI/blank node -> NULL (error)
        # ot LIKE '@%' means language-tagged -> rdf:langString
        # otherwise ot is the datatype URI
        return case(
            (type_col.is_(None), null()),
            (type_col.like("@%"), literal(str(RDF.langString))),
            else_=type_col,
        )
    # Constant handling
    if isinstance(arg_expr, Literal):
        if arg_expr.language:
            return literal(str(RDF.langString))
        return literal(str(arg_expr.datatype or XSD.string))
    # Non-literal constant -> NULL (error)
    return null()


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


def translate_pattern(
    node: CompValue, ctx: Context, engine: Engine = None
) -> QueryResult:
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

    # Subject
    if isinstance(s, Variable):
        if isinstance(p, Variable) and p == s:
            # Same variable in subject and predicate - handled in predicate section
            columns.append(ctx.table.c.s.label(str(s)))
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
        if isinstance(s, Variable) and p == s:
            conditions.append(ctx.table.c.p == ctx.table.c.s)
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
        if isinstance(s, Variable) and o == s:
            conditions.append(ctx.table.c.o == ctx.table.c.s)
            conditions.append(ctx.table.c.ot.is_(None))
        elif isinstance(p, Variable) and o == p:
            conditions.append(ctx.table.c.o == ctx.table.c.p)
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
        # Only project graph variable if NOT inside a GRAPH ?g pattern
        # (proper SPARQL scoping - variable bound after pattern evaluation)
        if isinstance(ctx.graph_term, Variable) and not ctx._inside_graph_var_pattern:
            columns.append(ctx.table.c.g.label(str(ctx.graph_term)))
        elif isinstance(ctx.graph_term, Variable):
            # Use internal name so it's NOT visible to BOUND etc.
            # _pattern_graph will rename it to the variable name after evaluation
            columns.append(ctx.table.c.g.label(INTERNAL_GRAPH_COLUMN))

    # Ensure valid SELECT clause (SQL requires at least one column)
    if not columns:
        columns = [literal(1).label(EMPTY_PROJECTION_MARKER)]

    query = select(*columns).select_from(ctx.table)
    if conditions:
        query = query.where(*conditions)
    return query


def _expand_path(s, path, o, ctx: Context) -> QueryResult:
    """Recursively expand a property path to SQL.

    Args:
        s: Subject term (Variable, URIRef, BNode, or Literal)
        path: Property path (URIRef or Path subclass)
        o: Object term (Variable, URIRef, BNode, or Literal)
        ctx: Translation context

    Returns:
        A single QueryResult representing the path pattern.
    """
    # Simple predicate - delegate to triple_to_query
    if isinstance(path, URIRef):
        return triple_to_query((s, path, o), ctx)

    # Dispatch by path type
    if isinstance(path, SequencePath):
        return _expand_sequence(s, path, o, ctx)
    if isinstance(path, MulPath):
        return _mulpath_to_cte(s, path, o, ctx)
    if isinstance(path, InvPath):
        return _expand_path(o, path.arg, s, ctx)
    if isinstance(path, NegatedPath):
        return _expand_negated_path(s, path, o, ctx)

    raise NotImplementedError(f"Path type {type(path)} not implemented")


def _expand_sequence(s, seq: SequencePath, o, ctx: Context) -> QueryResult:
    """Expand a sequence path (p/q/r) by joining recursive expansions."""
    temp_vars = [Variable(f"_seq_{id(seq)}_{i}") for i in range(len(seq.args) - 1)]
    nodes = [s] + temp_vars + [o]
    queries = [
        _expand_path(nodes[i], seq.args[i], nodes[i + 1], ctx)
        for i in range(len(seq.args))
    ]
    return _bgp_join_queries(queries, ctx)


def _expand_negated_path(s, negpath: NegatedPath, o, ctx: Context) -> QueryResult:
    """Expand negated path (!p) using NOT IN clause.

    Reuses triple_to_query with a placeholder predicate, adding NOT IN filter.
    """
    forward = [term_to_string(a) for a in negpath.args if isinstance(a, URIRef)]
    inverse = [
        term_to_string(a.arg)
        for a in negpath.args
        if isinstance(a, InvPath) and isinstance(a.arg, URIRef)
    ]

    # Placeholder predicate variable (filtered out in projection)
    pred_var = Variable(f"_negp_{id(negpath)}")

    def build(subj, obj, excluded):
        base = triple_to_query((subj, pred_var, obj), ctx)
        return base.where(~ctx.table.c.p.in_(excluded)) if excluded else base

    queries = (
        [build(s, o, forward)] if forward or not inverse else []
    ) + ([build(o, s, inverse)] if inverse else [])

    return queries[0] if len(queries) == 1 else union_all_queries(queries, s, o, ctx)


def expand_triple(triple: tuple, ctx: Context) -> List[QueryResult]:
    """Expand a triple pattern into a list of queries.

    For simple predicates, returns a single-element list.
    For path predicates, recursively expands and returns a single-element list.
    """
    s, p, o = triple

    if not isinstance(p, Path):
        return [triple_to_query(triple, ctx)]

    return [_expand_path(s, p, o, ctx)]


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

    # For p? (zero-or-one), we only want length 0 or 1, so skip recursion
    if mulpath.zero and not mulpath.more:
        # p? = zero-length UNION one-hop
        one_hop = _build_mulpath_result(
            base_query.cte(), s, o, s_value, o_value, o_type, ctx
        )
        zero_len = _zero_length_query(s, o, s_value, o_value, ctx)
        return union_all_queries([one_hop, zero_len], s, o, ctx)

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

    positive_paths = _build_mulpath_result(
        path_cte, s, o, s_value, o_value, o_type, ctx
    )

    # For p* (zero-or-more), UNION with zero-length paths
    if mulpath.zero:
        zero_len = _zero_length_query(s, o, s_value, o_value, ctx)
        return union_all_queries([positive_paths, zero_len], s, o, ctx)

    return positive_paths


def _build_mulpath_result(
    path_cte, s, o, s_value, o_value, o_type, ctx: Context
) -> Select:
    """Build the final SELECT from a MulPath CTE with appropriate filters."""
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

    return final_query.where(and_(*conditions)) if conditions else final_query


def _zero_length_query(s, o, s_value, o_value, ctx: Context) -> Optional[Select]:
    """Generate query for zero-length path matches (node = node)."""
    # Both bound - only match if equal
    if s_value is not None and o_value is not None:
        return (
            _literal_row({str(s): s_value, str(o): o_value})
            if s_value == o_value
            else None
        )

    # Output variables: (name, is_node_var) - node vars get the value, graph var gets NULL
    out_vars = [
        (str(v), v in (s, o)) for v in [s, o, ctx.graph_term] if isinstance(v, Variable)
    ]
    if not out_vars:
        return None

    # One bound - literal row
    if (bound := s_value or o_value) is not None:
        return _literal_row(
            {name: bound if is_node else None for name, is_node in out_vars}
        )

    # Both unbound - select from all graph nodes
    nodes = _all_graph_nodes(ctx).subquery()
    return select(
        *[
            nodes.c.node.label(name) if is_node else literal(None).label(name)
            for name, is_node in out_vars
        ]
    ).select_from(nodes)


def _all_graph_nodes(ctx: Context):
    """Return union of all subjects and non-literal objects in the graph."""
    t, g = ctx.table, ctx.graph_filter
    subjects = select(t.c.s.label("node"))
    objects = select(t.c.o.label("node")).where(t.c.ot.is_(None))
    if g:
        subjects, objects = subjects.where(g), objects.where(g)
    return union(subjects, objects)


def _literal_row(columns: dict) -> Select:
    """Generate a single-row SELECT with literal values."""
    return select(*[literal(v).label(k) for k, v in columns.items()])


def union_all_queries(queries: list, s, o, ctx: Context) -> Select:
    """UNION ALL multiple queries, filtering out None values."""
    queries = [q for q in queries if q is not None]
    if not queries:
        raise ValueError("No valid queries to union")
    if len(queries) == 1:
        return queries[0]

    result = queries[0]
    for q in queries[1:]:
        result = result.union_all(q)

    return result


# =============================================================================
# Join Utilities
# =============================================================================


def join_queries(queries: list, ctx: Context) -> QueryResult:
    """Natural join multiple queries on common variables (SPARQL semantics).

    Uses NULL-tolerant join conditions where unbound variables match anything.
    This is needed for cross-pattern joins (e.g., between UNION branches).
    """
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


def _bgp_join_queries(queries: list, ctx: Context) -> QueryResult:
    """Join queries within a BGP using strict INNER JOINs.

    Within a Basic Graph Pattern, all triple patterns must match for a solution
    to exist. Variables are never NULL within a successful BGP match, so we can
    use efficient INNER JOINs with strict equality instead of NULL-tolerant joins.

    This produces SQL like:
        cte1 JOIN cte2 ON cte1.x = cte2.x JOIN cte3 ON cte2.y = cte3.y
    """
    if len(queries) == 1:
        return queries[0]

    ctes = [as_cte(q) for q in queries]

    # Order CTEs to minimize intermediate result sizes:
    ordered_ctes = _order_ctes_for_join(ctes)

    # Start with first CTE as base
    base_cte = ordered_ctes[0]
    joined_cols = set(base_cte.c.keys())

    # Track column sources for final projection
    col_sources = {name: base_cte.c[name] for name in joined_cols}

    # Build the join expression incrementally
    join_expr = base_cte

    for cte in ordered_ctes[1:]:
        cte_cols = set(cte.c.keys())
        common = joined_cols & cte_cols

        if common:
            # Build strict equality join conditions
            on_clause = and_(*[col_sources[col] == cte.c[col] for col in common])
        else:
            # No common columns - cross join (rare but valid)
            on_clause = true()

        # Extend the join
        join_expr = join_expr.join(cte, on_clause)

        # Track new columns (first occurrence wins for projection)
        for name in cte_cols - joined_cols:
            col_sources[name] = cte.c[name]

        joined_cols |= cte_cols

    # Build final SELECT with all columns
    final_cols = [col_sources[name].label(name) for name in col_sources]

    return select(*final_cols).select_from(join_expr)


def _order_ctes_for_join(ctes: list) -> list:
    """Order CTEs to minimize intermediate result sizes during joins.

    Heuristic: Start with CTEs that have fewer columns (likely more selective),
    then greedily add CTEs that share variables with the already-joined set.
    """
    if len(ctes) <= 2:
        return ctes

    remaining = list(ctes)
    # Start with the CTE that has the fewest columns (most constrained)
    remaining.sort(key=lambda c: len(c.c.keys()))
    ordered = [remaining.pop(0)]
    joined_cols = set(ordered[0].c.keys())

    while remaining:
        # Find CTE with most overlap with joined columns
        best_idx = 0
        best_overlap = -1

        for i, cte in enumerate(remaining):
            overlap = len(joined_cols & set(cte.c.keys()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        chosen = remaining.pop(best_idx)
        ordered.append(chosen)
        joined_cols |= set(chosen.c.keys())

    return ordered


# =============================================================================
# Pattern Handlers
# =============================================================================


@pattern_handler("BGP")
def _pattern_bgp(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Basic Graph Pattern.

    Uses strict INNER JOINs since all patterns in a BGP must match.
    Variables are never NULL within a successful BGP match.
    """
    triples = node["triples"]

    if not triples:
        # Empty BGP handling - returns one empty solution if conditions met
        if ctx.graph_term is None:
            if not ctx.graph_aware:
                return select(literal(1).label(EMPTY_PROJECTION_MARKER))
            return (
                select(literal(1).label(EMPTY_PROJECTION_MARKER))
                .select_from(ctx.table)
                .where(ctx.table.c.g.is_(None))
                .limit(1)
            )
        if isinstance(ctx.graph_term, Variable):
            # Inside GRAPH ?g: use internal column name so var is NOT in scope
            # Renamed to the variable by _pattern_graph after inner evaluation
            return (
                select(ctx.table.c.g.label(INTERNAL_GRAPH_COLUMN))
                .where(ctx.table.c.g.isnot(None))
                .distinct()
            )
        # Specific graph: check existence
        return (
            select(literal(1).label(EMPTY_PROJECTION_MARKER))
            .select_from(ctx.table)
            .where(ctx.table.c.g == str(ctx.graph_term))
            .limit(1)
        )

    queries = [q for triple in triples for q in expand_triple(triple, ctx)]
    return _bgp_join_queries(queries, ctx)


@pattern_handler("Graph")
def _pattern_graph(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate GRAPH pattern with proper variable scoping.

    SPARQL semantics: the graph variable in GRAPH ?g { P } is NOT in scope
    inside P. It's only bound after P is evaluated, then unified with any
    bindings of the same variable from inside P.
    """
    if not ctx.graph_aware:
        raise ValueError("Graph patterns require graph_aware=True")

    graph_term = node["term"]

    if not isinstance(graph_term, Variable):
        return translate_pattern(node["p"], ctx.with_graph(graph_term), engine)

    # Translate inner pattern with graph variable hidden (proper scoping)
    inner_ctx = ctx.with_graph(graph_term, inside_var_pattern=True)
    cte = as_cte(translate_pattern(node["p"], inner_ctx, engine))
    graph_var_name = str(graph_term)

    # Keep all columns except internal graph col and any same-named binding
    cols = [c for c in cte.c if c.key not in (INTERNAL_GRAPH_COLUMN, graph_var_name)]
    cols.append(cte.c[INTERNAL_GRAPH_COLUMN].label(graph_var_name))
    query = select(*cols).select_from(cte)

    # Unify if inner pattern also bound this variable (e.g., GRAPH ?g { ?g ?p ?o })
    if graph_var_name in cte.c.keys():
        query = query.where(
            or_(
                cte.c[graph_var_name].is_(None),
                cte.c[graph_var_name] == cte.c[INTERNAL_GRAPH_COLUMN],
            )
        )

    return query


@pattern_handler("Project")
def _pattern_project(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Project (SELECT) operation.

    For empty projections, results contain EMPTY_PROJECTION_MARKER ('__empty__')
    since SQL requires at least one column.
    """
    project_vars = node["PV"]
    base_query = translate_pattern(node["p"], ctx, engine)
    var_names = [str(var) for var in project_vars]

    # No-op if columns already match
    if col_names(base_query) == set(var_names):
        return base_query

    if not project_vars:
        cte = as_cte(base_query)
        return select(literal(1).label(EMPTY_PROJECTION_MARKER)).select_from(cte)

    # For CTE, wrap with select; for Select, use with_only_columns to preserve ORDER BY
    if isinstance(base_query, CTE):
        return select(*_project_columns(cols(base_query), var_names)).select_from(
            base_query
        )

    return base_query.with_only_columns(*_project_columns(cols(base_query), var_names))


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
    """Translate Distinct operation with term identity semantics.

    SPARQL DISTINCT compares by term identity (value + datatype), not just value.
    We include _ot_* (object type) columns in the comparison, then strip them.
    """
    inner_node = node["p"]

    # If inner node is Project, include _ot_* columns for term identity comparison
    if hasattr(inner_node, "name") and inner_node.name == "Project":
        var_names = [str(var) for var in inner_node["PV"]]
        bgp_query = translate_pattern(inner_node["p"], ctx, engine)
        distinct_cols = _project_columns(cols(bgp_query), var_names, include_ot=True)

        # DISTINCT on value + type columns, then strip _ot_* in final projection
        cte = as_cte(_apply_distinct(bgp_query, distinct_cols))
        user_cols = [cte.c[v] for v in var_names if v in cte.c.keys()]
        return select(*user_cols).select_from(cte)

    # Non-Project: simple DISTINCT
    return _apply_distinct(translate_pattern(inner_node, ctx, engine))


@pattern_handler("Reduced")
def _pattern_reduced(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate SELECT REDUCED patterns.

    SPARQL REDUCED permits but does not require duplicate elimination.
    We treat it as a no-op since that is what the W3C test suite expects.
    """
    return translate_pattern(node["p"], ctx, engine)


@pattern_handler("Extend")
def _pattern_extend(node: CompValue, ctx: Context, engine) -> QueryResult:
    """Translate Extend (BIND) operation."""
    inner_query = translate_pattern(node["p"], ctx, engine)
    cte = as_cte(inner_query)
    var_to_col = cols(cte)

    sql_expr = translate_expr(node["expr"], var_to_col, engine)

    # Boolean expressions must return XSD boolean strings ("true"/"false") when projected
    if _is_boolean_expr(node["expr"]):
        sql_expr = _bool_to_xsd_string(sql_expr)

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
        if isinstance(base_query, SelectBase):
            # Select or CompoundSelect (from UNION) - return directly
            return base_query
        # CTE - wrap in a SELECT
        return select(*base_query.c).select_from(base_query)

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

_APP_NAME = "sparql2sql"


def create_databricks_engine(
    server_hostname: str, http_path: str, access_token: str, **engine_kwargs
) -> Engine:
    """Create a SQLAlchemy engine for Databricks.

    The engine is configured with _user_agent_entry=sparql2sql so that queries
    are identifiable in database logging.
    """
    engine_uri = (
        f"databricks://token:{access_token}@{server_hostname}"
        f"?http_path={http_path}&_user_agent_entry={_APP_NAME}"
    )
    return create_engine(engine_uri, **engine_kwargs)


def create_postgres_engine(connection_url: str, **engine_kwargs) -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL.

    The engine is configured with application_name=sparql2sql so that queries
    are identifiable in database logging (e.g., in pg_stat_activity).

    Args:
        connection_url: PostgreSQL connection URL, e.g.,
            "postgresql://user:password@localhost/dbname"
        **engine_kwargs: Additional arguments passed to create_engine()
    """
    connect_args = engine_kwargs.pop("connect_args", {})
    connect_args.setdefault("application_name", _APP_NAME)
    return create_engine(connection_url, connect_args=connect_args, **engine_kwargs)


def create_sqlite_engine(
    connection_url: str = "sqlite:///:memory:", **engine_kwargs
) -> Engine:
    """Create a SQLAlchemy engine for SQLite with REGEXP support.

    This registers a Python-based regexp function that enables the SPARQL
    REGEX function to work with SQLite databases.

    Args:
        connection_url: SQLite connection URL, e.g.,
            "sqlite:///path/to/db.sqlite" or "sqlite:///:memory:"
        **engine_kwargs: Additional arguments passed to create_engine()
    """
    from sqlalchemy import event

    def _sqlite_regexp(pattern: str, text: str) -> bool:
        """SQLite regexp user function implementation using Python's re module."""
        if text is None:
            return False
        return re.search(pattern, text) is not None

    engine = create_engine(connection_url, **engine_kwargs)

    @event.listens_for(engine, "connect")
    def _register_regexp(dbapi_connection, connection_record):
        """Register the regexp function for each new connection."""
        dbapi_connection.create_function("regexp", 2, _sqlite_regexp)

    return engine
