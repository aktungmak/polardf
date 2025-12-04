import itertools
from typing import List, Optional, Union

from rdflib import Literal, URIRef
from rdflib.paths import Path, SequencePath, MulPath
from rdflib.plugins.sparql import parser, algebra
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import Variable
from sqlalchemy import MetaData, Table, Column, String, select, create_engine, true, Engine, CursorResult, quoted_name, \
    CTE, Select, and_, null
from sqlalchemy.sql import column


def term_to_string(term) -> Optional[str]:
    """Convert an RDF term (URIRef or Literal) to its string representation.
    
    Returns None if the term is not a URIRef or Literal (e.g., a Variable).
    """
    if isinstance(term, URIRef):
        return f"<{term}>"
    elif isinstance(term, Literal):
        return str(term)
    return None


def create_databricks_engine(
        server_hostname: str,
        http_path: str,
        access_token: str,
        **engine_kwargs
) -> Engine:
    engine_uri = f"databricks://token:{access_token}@{server_hostname}?http_path={http_path}"
    return create_engine(engine_uri, **engine_kwargs)


class AlgebraTranslator:
    """Translates SPARQL algebra expressions to SQLAlchemy queries.
    The assumption is that the triples are stored in a table with columns s, p, and o."""

    def __init__(
            self,
            engine: Engine,
            table_name: str = "triples",
            create_table: bool = False
    ):
        self.engine = engine
        self.metadata = MetaData()
        self._cte_counter = itertools.count()

        # Define the triple table structure
        self.table = Table(
            quoted_name(table_name, quote=False),
            self.metadata,
            Column('s', String, nullable=False, primary_key=True),
            Column('p', String, nullable=False, primary_key=True),
            Column('o', String, nullable=False, primary_key=True),
        )

        if create_table:
            self.metadata.create_all(self.engine)

    def execute(self, sparql_query: str) -> CursorResult:
        """Translate a SPARQL query and execute it."""
        sql_query = self.translate(sparql_query)
        print("translated:\n", sql_query)
        with self.engine.connect() as conn:
            return conn.execute(sql_query)

    def translate(self, sparql_string: str):
        """Translate a SPARQL query string to a SQLAlchemy query."""
        query_tree = parser.parseQuery(sparql_string)
        query_algebra = algebra.translateQuery(query_tree)
        print("algebra:")
        algebra.pprintAlgebra(query_algebra)
        query_algebra = query_algebra.algebra
        if hasattr(query_algebra, 'name'):
            if query_algebra.name == "SelectQuery":
                return self._translate_select_query(query_algebra)

        raise NotImplementedError(f"Algebra translation not implemented for {query_algebra}")

    def _translate_select_query(self, select_query: CompValue):
        base_query = self._translate_pattern(select_query["p"])

        # If the base query is a CTE, we need to select from it
        if isinstance(base_query, CTE):
            return select(*base_query.c).select_from(base_query)

        # Otherwise, the base query is already a complete select, just return it
        return base_query

    def _translate_pattern(self, pattern):
        """Translate different pattern types."""
        if hasattr(pattern, 'name'):
            if pattern.name == "BGP":
                return self._translate_bgp(pattern)
            elif pattern.name == "Project":
                return self._translate_project(pattern)
            elif pattern.name == "LeftJoin":
                return self._translate_left_join(pattern)
            elif pattern.name == "Distinct":
                return self._translate_distinct(pattern)
            else:
                raise NotImplementedError(f"Pattern type {pattern.name} not implemented")
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
            raise NotImplementedError(f"MulPath with path type {type(mulpath.path)} not implemented")

        # Convert bound values to strings for comparison
        s_value = term_to_string(s)
        o_value = term_to_string(o)

        # Base case: all direct edges with the predicate
        base_query = select(
            self.table.c.s,
            self.table.c.p,
            self.table.c.o
        ).where(self.table.c.p == pred_value)

        # Create recursive CTE with unique name
        cte_name = f"path_cte_{next(self._cte_counter)}"
        path_cte = base_query.cte(name=cte_name, recursive=True)

        # Recursive case: extend paths forward
        recursive_query = select(
            path_cte.c.s,  # Keep original starting node
            self.table.c.p,
            self.table.c.o  # Extend to new destinations
        ).select_from(
            path_cte.join(self.table, path_cte.c.o == self.table.c.s)
        ).where(self.table.c.p == pred_value)

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

        # Select from all CTEs with WHERE conditions for common columns
        return select(*ctes).select_from(*ctes).where(*conditions)

    def _translate_project(self, project):
        """Translate a Project (SELECT) operation."""
        project_vars = project["PV"]
        pattern = project["p"]

        # Translate the inner pattern
        base_query = self._translate_pattern(pattern)

        # Get column names from base query
        if isinstance(base_query, CTE):
            base_cols = set(base_query.c.keys())
        elif hasattr(base_query, 'selected_columns'):
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
            join_conditions = [left_cte.c[col] == right_cte.c[col] for col in common_cols]

            # All columns from both sides
            all_columns = list(left_cte.c) + [col for name, col in right_cte.c.items() if name not in common_cols]

            # LEFT JOIN - return as CTE since we need to join - use and_() to combine multiple conditions
            return select(*all_columns).select_from(
                left_cte.outerjoin(right_cte, and_(*join_conditions))
            ).cte()
        else:
            # If no common variables, it's a cartesian product with optional semantics
            all_columns = list(left_cte.c) + list(right_cte.c)
            return select(*all_columns).select_from(
                left_cte.outerjoin(right_cte, true())
            ).cte()


# Example usage
if __name__ == "__main__":
    sparql_query = """
    PREFIX dbx: <http://www.databricks.com/ontology/>
    SELECT *
    WHERE {
    ?cat a dbx:Catalog .
    ?tab dbx:tableInSchema/dbx:inCatalog ?cat .
    }"""

    engine = create_databricks_engine(
        server_hostname="e2-demo-field-eng.cloud.databricks.com",
        http_path="/sql/1.0/warehouses/8baced1ff014912d",
        access_token=""
    )

    translator = AlgebraTranslator(
        engine=engine,
        table_name="users.joshua_green.triples"
    )

    result = translator.execute(sparql_query)
    print("Results:", result.fetchall())

    engine.dispose()
