import itertools
from rdflib import Literal, URIRef
from rdflib.paths import Path, SequencePath, MulPath
from rdflib.plugins.sparql import parser, algebra
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import Variable
from sqlalchemy import MetaData, Table, Column, String, select, create_engine, true, Engine, CursorResult, quoted_name, \
    CTE, Select, and_
from sqlalchemy.sql import column


def create_databricks_engine(
        server_hostname: str,
        http_path: str,
        access_token: str,
        **engine_kwargs
) -> Engine:
    engine_uri = f"databricks://token:{access_token}@{server_hostname}?http_path={http_path}"
    return create_engine(engine_uri, **engine_kwargs)


class AlgebraTranslator:
    """Translates SPARQL algebra expressions to SQLAlchemy queries."""

    def __init__(
            self,
            engine: Engine,
            table_name: str = "triples",
            create_table: bool = False
    ):
        self.engine = engine
        self.metadata = MetaData()

        # Define the triple table structure
        self.table = Table(
            quoted_name(table_name, quote=False),
            self.metadata,
            Column('s', String, nullable=False, primary_key=True),
            Column('p', String, nullable=False, primary_key=True),
            Column('o', String, nullable=False, primary_key=True),
        )

        # Only create table if explicitly requested
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
        pattern = select_query["p"]
        base_query = self._translate_pattern(pattern)

        # If the base query is a CTE, we need to select from it
        if isinstance(base_query, CTE):
            assert False
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
            else:
                raise NotImplementedError(f"Pattern type {pattern.name} not implemented")
        else:
            raise ValueError(f"Unknown pattern type: {pattern}")

    def _translate_bgp(self, bgp):
        """Translate a Basic Graph Pattern to a query."""
        triples = bgp["triples"]

        # Expand any path predicates
        expanded_triples = []
        for triple in triples:
            expanded_triples.extend(self._expand_path_triple(triple))

        if len(expanded_triples) == 1:
            # Single triple pattern
            return self._triple_to_query(expanded_triples[0])
        else:
            # Multiple triples - need to join them
            # TODO it might be simpler to save the joining until later
            queries = [self._triple_to_query(triple) for triple in expanded_triples]
            return self._join_queries(queries)

    def _expand_path_triple(self, triple) -> list[tuple]:
        """Expand a triple with a path predicate into multiple triples."""
        s, p, o = triple

        if not isinstance(p, Path):
            return [triple]

        # Handle SequencePath (e.g., <a>/<b>/<c>)
        if isinstance(p, SequencePath):
            temp_vars = [Variable(f"_path_{id(p)}_{i}") for i in range(len(p.args) - 1)]
            subjects = [s] + temp_vars
            predicates = p.args
            objects = temp_vars + [o]
            return list(zip(subjects, predicates, objects))

        elif isinstance(p, MulPath):
            # TODO implement this using a recursive query
            NotImplementedError("MulPath not yet implemented")
            pass
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
        elif isinstance(s, URIRef):
            conditions.append(self.table.c.s == f"<{s}>")
        else:
            raise NotImplementedError(f"Subject type {type(s)} not implemented")

        # Predicate  
        if isinstance(p, Variable):
            columns.append(self.table.c.p.label(str(p)))
        elif isinstance(p, URIRef):
            conditions.append(self.table.c.p == f"<{p}>")
        else:
            raise NotImplementedError(f"Predicate type {type(p)} not implemented")

        # Object
        if isinstance(o, Variable):
            columns.append(self.table.c.o.label(str(o)))
        elif isinstance(o, URIRef):
            conditions.append(self.table.c.o == f"<{o}>")
        elif isinstance(o, Literal):
            conditions.append(self.table.c.o == str(o))
        else:
            raise NotImplementedError(f"Object type {type(o)} not implemented")

        query = select(*columns).select_from(self.table)
        if conditions:
            query = query.where(*conditions)

        return query

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

        # Apply projection - use subquery to avoid unnecessary CTE
        var_columns = [column(str(var)) for var in project_vars]
        return select(*var_columns).select_from(base_query.subquery())

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
