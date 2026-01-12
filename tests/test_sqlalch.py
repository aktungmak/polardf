import unittest
from unittest.mock import Mock, patch, MagicMock
from rdflib import XSD
from sqlalchemy import create_engine, Engine
from sqlalchemy.sql.selectable import Select, Subquery, CTE
from rdfdf.sqlalch import AlgebraTranslator, create_databricks_engine


class TestAlgebraTranslator(unittest.TestCase):
    """Tests for the AlgebraTranslator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create an in-memory SQLite database engine for testing
        self.engine = create_engine("sqlite:///:memory:")
        # Create the translator with a test triples table
        self.translator = AlgebraTranslator(
            self.engine, table_name="triples", create_table=True
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Dispose of the engine to clean up resources
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_translate_path_expression_query(self):
        """Test translation of a SPARQL query with path expressions.

        This test uses the example query from sqlalch.py that includes:
        - A simple triple pattern (?cat a dbx:Catalog)
        - A sequence path pattern (?tab dbx:tableInSchema/dbx:inCatalog ?cat)
        """
        sparql_query = """
        PREFIX dbx: <http://www.databricks.com/ontology/>
        SELECT *
        WHERE {
            ?cat a dbx:Catalog .
            ?tab dbx:tableInSchema/dbx:inCatalog ?cat .
        }"""

        # Translate the query
        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Check that we have the expected projection columns (cat and tab)
        selected_column_names = {col.key for col in sql_query.selected_columns}
        self.assertEqual(selected_column_names, {"cat", "tab"})

        # Check that the query uses a subquery (for the joins)
        froms = sql_query.get_final_froms()
        self.assertEqual(len(froms), 1)

        # The FROM should be a subquery
        subquery = froms[0]
        self.assertIsInstance(subquery, Subquery)

        # The subquery should have 3 CTEs involved (one for each triple pattern)
        # We can verify this by checking the subquery's element
        inner_select = subquery.element
        self.assertIsInstance(inner_select, Select)

        # Check that there are WHERE conditions in the subquery (for the joins)
        self.assertIsNotNone(inner_select.whereclause)

        # Verify the query is executable (doesn't raise an error when compiled)
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_translate_optional_query(self):
        """Test translation of a SPARQL query with OPTIONAL (LEFT JOIN).

        This test verifies that OPTIONAL patterns are correctly translated
        to LEFT JOIN operations in SQL.
        """
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT *
        WHERE {
            ?person ex:name ?name .
            OPTIONAL { ?person ex:age ?age }
        }"""

        # Translate the query
        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Check that we have the expected projection columns (person, name, age)
        selected_column_names = {col.key for col in sql_query.selected_columns}
        self.assertEqual(selected_column_names, {"person", "name", "age"})

        # Verify the query is executable
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_translate_order_by_asc(self):
        """Test translation of ORDER BY with ascending order."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?s ?o
        WHERE { ?s ex:p ?o }
        ORDER BY ?o
        """

        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Check that ORDER BY clause exists
        self.assertIsNotNone(sql_query._order_by_clauses)
        self.assertEqual(len(sql_query._order_by_clauses), 1)

        # Verify the query is executable
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
            # Check that ORDER BY appears in the compiled SQL
            sql_str = str(compiled)
            self.assertIn("ORDER BY", sql_str)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_translate_order_by_desc(self):
        """Test translation of ORDER BY with descending order."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?s ?o
        WHERE { ?s ex:p ?o }
        ORDER BY DESC(?o)
        """

        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Check that ORDER BY clause exists
        self.assertIsNotNone(sql_query._order_by_clauses)
        self.assertEqual(len(sql_query._order_by_clauses), 1)

        # Verify the query is executable
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
            # Check that ORDER BY DESC appears in the compiled SQL
            sql_str = str(compiled)
            self.assertIn("ORDER BY", sql_str)
            self.assertIn("DESC", sql_str)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_translate_order_by_multiple(self):
        """Test translation of ORDER BY with multiple sort keys."""
        sparql_query = """
        SELECT ?s ?p ?o
        WHERE { ?s ?p ?o }
        ORDER BY ?s DESC(?o)
        """

        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Check that ORDER BY clause has two elements
        self.assertIsNotNone(sql_query._order_by_clauses)
        self.assertEqual(len(sql_query._order_by_clauses), 2)

        # Verify the query is executable
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")


class TestAlgebraTranslatorExecute(unittest.TestCase):
    """Tests for executing queries against sample data."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        # Create an in-memory SQLite database engine for testing
        self.engine = create_engine("sqlite:///:memory:")
        # Create the translator with a test triples table
        self.translator = AlgebraTranslator(
            self.engine, table_name="triples", create_table=True
        )

        # Insert sample data
        with self.engine.connect() as conn:
            # Data for path expression tests
            # Relationships: alice -> bob -> charlie (parent chain)
            # ot=None means the object is an IRI
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://example.org/alice",
                        "p": "http://example.org/parent",
                        "o": "http://example.org/bob",
                        "ot": None,
                    },
                    {
                        "s": "http://example.org/bob",
                        "p": "http://example.org/parent",
                        "o": "http://example.org/charlie",
                        "ot": None,
                    },
                ],
            )

            # Data for OPTIONAL tests
            # alice has name "Alice"
            # bob has name "Bob" and age "30"
            # charlie has name "Charlie" (no age)
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://example.org/alice",
                        "p": "http://example.org/name",
                        "o": "Alice",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://example.org/bob",
                        "p": "http://example.org/name",
                        "o": "Bob",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://example.org/bob",
                        "p": "http://example.org/age",
                        "o": "30",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://example.org/charlie",
                        "p": "http://example.org/name",
                        "o": "Charlie",
                        "ot": str(XSD.string),
                    },
                ],
            )
            conn.commit()

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_execute_path_expression_query(self):
        """Test executing a SPARQL query with path expressions against sample data.

        This query finds all grandparent relationships using a path expression.
        Expected result: alice's grandparent is charlie (alice -> bob -> charlie)
        """
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT *
        WHERE {
            ?person ex:parent/ex:parent ?grandparent .
        }"""

        # Execute the query
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Verify we got exactly one result
        self.assertEqual(len(rows), 1)

        # Verify the result contains alice and charlie
        row = rows[0]
        self.assertEqual(row.person, "http://example.org/alice")
        self.assertEqual(row.grandparent, "http://example.org/charlie")

    def test_execute_optional_query(self):
        """Test executing a SPARQL query with OPTIONAL against sample data.

        This query finds people with their names, and optionally their ages.
        Expected results:
        - alice with name "Alice" (no age)
        - bob with name "Bob" and age "30"
        - charlie with name "Charlie" (no age)
        """
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT *
        WHERE {
            ?person ex:name ?name .
            OPTIONAL { ?person ex:age ?age }
        }"""

        # Execute the query
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Verify we got three results (alice, bob, charlie)
        self.assertEqual(len(rows), 3)

        # Convert to a dict for easier checking
        results_dict = {row.person: {"name": row.name, "age": row.age} for row in rows}

        # Verify alice (has name, no age)
        self.assertIn("http://example.org/alice", results_dict)
        self.assertEqual(results_dict["http://example.org/alice"]["name"], "Alice")
        self.assertIsNone(results_dict["http://example.org/alice"]["age"])

        # Verify bob (has both name and age)
        self.assertIn("http://example.org/bob", results_dict)
        self.assertEqual(results_dict["http://example.org/bob"]["name"], "Bob")
        self.assertEqual(results_dict["http://example.org/bob"]["age"], "30")

        # Verify charlie (has name, no age)
        self.assertIn("http://example.org/charlie", results_dict)
        self.assertEqual(results_dict["http://example.org/charlie"]["name"], "Charlie")
        self.assertIsNone(results_dict["http://example.org/charlie"]["age"])

    def test_execute_same_variable_triple_pattern(self):
        """Test executing a query where the same variable appears multiple times.

        When a variable appears in multiple positions (e.g., ?x ?x ?x), the query
        should only return triples where those positions have equal values.
        """
        # Insert a triple where s=p=o for testing (ot=None means object is an IRI)
        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://example.org/same",
                        "p": "http://example.org/same",
                        "o": "http://example.org/same",
                        "ot": None,
                    },
                ],
            )
            conn.commit()

        # Test ?x ?x ?x - should only match triples where s=p=o
        sparql_query = """
        SELECT ?x WHERE { ?x ?x ?x }
        """
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should only return the one triple where s=p=o
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].x, "http://example.org/same")

        # Test ?x ?p ?x - should only match triples where s=o
        sparql_query2 = """
        SELECT ?x ?p WHERE { ?x ?p ?x }
        """
        result2 = self.translator.execute(sparql_query2)
        rows2 = result2.fetchall()

        # Should only return the triple where s=o
        self.assertEqual(len(rows2), 1)
        self.assertEqual(rows2[0].x, "http://example.org/same")

    def test_execute_empty_bgp(self):
        """Test executing a query with an empty BGP (no triple patterns).

        An empty BGP represents a single solution with no bindings,
        so it should return exactly one row.
        """
        sparql_query = "SELECT * WHERE {}"

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Empty BGP should return exactly one row (single solution with no bindings)
        self.assertEqual(len(rows), 1)

    def test_execute_empty_bgp_with_projection(self):
        """Test executing a query that projects a variable from an empty BGP.

        In SPARQL, SELECT ?x WHERE {} should return one row with ?x unbound (NULL).
        """
        sparql_query = "SELECT ?x WHERE {}"

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should return one row with x = NULL (unbound)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].x)

    def test_execute_projection_nonexistent_variable(self):
        """Test projecting a variable that doesn't exist in the pattern.

        SELECT ?z WHERE { ?x ex:knows ?y } should return rows with z = NULL.
        """
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?z WHERE { ?x ex:parent ?y }
        """

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should return rows (one per match) with z = NULL
        self.assertEqual(len(rows), 2)  # alice->bob, bob->charlie
        for row in rows:
            self.assertIsNone(row.z)

    def test_execute_projection_mixed_variables(self):
        """Test projecting a mix of existing and non-existing variables.

        SELECT ?person ?missing WHERE { ?person ex:name ?name } should return
        rows with person bound and missing = NULL.
        """
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person ?missing WHERE { ?person ex:name ?name }
        """

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should return 3 rows (alice, bob, charlie have names)
        self.assertEqual(len(rows), 3)

        persons = set()
        for row in rows:
            self.assertIsNotNone(row.person)
            self.assertIsNone(row.missing)
            persons.add(row.person)

        # Verify all three people are returned
        self.assertEqual(
            persons,
            {
                "http://example.org/alice",
                "http://example.org/bob",
                "http://example.org/charlie",
            },
        )

    def test_execute_distinct(self):
        """Test that SELECT DISTINCT removes duplicate rows."""
        # Without DISTINCT: returns one row per triple (duplicates for ?person)
        result = self.translator.execute("SELECT ?person WHERE { ?person ?p ?o }")
        rows = result.fetchall()
        persons = [row.person for row in rows]
        self.assertGreater(
            len(persons), len(set(persons)), "Should have duplicates without DISTINCT"
        )

        # With DISTINCT: each person appears exactly once
        result = self.translator.execute(
            "SELECT DISTINCT ?person WHERE { ?person ?p ?o }"
        )
        rows = result.fetchall()
        persons = [row.person for row in rows]
        self.assertEqual(
            len(persons), len(set(persons)), "DISTINCT should eliminate duplicates"
        )

    def test_execute_order_by_asc(self):
        """Test ORDER BY ascending returns results in correct order."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?name
        WHERE { ?person ex:name ?name }
        ORDER BY ?name
        """

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()
        names = [row.name for row in rows]

        # Should be sorted alphabetically: Alice, Bob, Charlie
        self.assertEqual(names, ["Alice", "Bob", "Charlie"])

    def test_execute_order_by_desc(self):
        """Test ORDER BY descending returns results in correct order."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?name
        WHERE { ?person ex:name ?name }
        ORDER BY DESC(?name)
        """

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()
        names = [row.name for row in rows]

        # Should be sorted reverse alphabetically: Charlie, Bob, Alice
        self.assertEqual(names, ["Charlie", "Bob", "Alice"])

    def test_execute_order_by_multiple(self):
        """Test ORDER BY with multiple sort keys."""
        # Insert additional data with duplicate predicates for sorting test
        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://example.org/alice",
                        "p": "http://example.org/score",
                        "o": "100",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://example.org/bob",
                        "p": "http://example.org/score",
                        "o": "100",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://example.org/charlie",
                        "p": "http://example.org/score",
                        "o": "50",
                        "ot": str(XSD.string),
                    },
                ],
            )
            conn.commit()

        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person ?score
        WHERE { ?person ex:score ?score }
        ORDER BY ?score ?person
        """

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should be sorted by score first, then by person
        # score=100: alice, bob (alphabetically)
        # score=50: charlie
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].score, "100")
        self.assertEqual(rows[0].person, "http://example.org/alice")
        self.assertEqual(rows[1].score, "100")
        self.assertEqual(rows[1].person, "http://example.org/bob")
        self.assertEqual(rows[2].score, "50")
        self.assertEqual(rows[2].person, "http://example.org/charlie")

    def test_execute_limit(self):
        """Test LIMIT returns correct number of results."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?name
        WHERE { ?person ex:name ?name }
        LIMIT 2
        """

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should only return 2 results
        self.assertEqual(len(rows), 2)

    def test_execute_offset(self):
        """Test OFFSET skips correct number of results."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?name
        WHERE { ?person ex:name ?name }
        ORDER BY ?name
        OFFSET 1
        """

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()
        names = [row.name for row in rows]

        # Should skip Alice, return Bob and Charlie
        self.assertEqual(len(rows), 2)
        self.assertEqual(names, ["Bob", "Charlie"])

    def test_execute_limit_offset(self):
        """Test LIMIT and OFFSET together."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?name
        WHERE { ?person ex:name ?name }
        ORDER BY ?name
        LIMIT 1
        OFFSET 1
        """

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()
        names = [row.name for row in rows]

        # Should skip Alice (offset 1), then return only 1 result (Bob)
        self.assertEqual(len(rows), 1)
        self.assertEqual(names, ["Bob"])


class TestAlgebraTranslatorMulPath(unittest.TestCase):
    """Tests for MulPath translation and execution."""

    def setUp(self):
        """Set up test fixtures with path data."""
        # Create an in-memory SQLite database engine for testing
        self.engine = create_engine("sqlite:///:memory:")
        # Create the translator with a test triples table
        self.translator = AlgebraTranslator(
            self.engine, table_name="triples", create_table=True
        )

        # Insert sample data for path traversal tests
        with self.engine.connect() as conn:
            # Create a path: start -> node1 -> node2 -> end
            # Also: start -> alt -> end (shorter path)
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://example.org/start",
                        "p": "http://example.org/path_pred",
                        "o": "http://example.org/node1",
                        "ot": None,
                    },
                    {
                        "s": "http://example.org/node1",
                        "p": "http://example.org/path_pred",
                        "o": "http://example.org/node2",
                        "ot": None,
                    },
                    {
                        "s": "http://example.org/node2",
                        "p": "http://example.org/path_pred",
                        "o": "http://example.org/end",
                        "ot": None,
                    },
                    {
                        "s": "http://example.org/start",
                        "p": "http://example.org/path_pred",
                        "o": "http://example.org/alt",
                        "ot": None,
                    },
                    {
                        "s": "http://example.org/alt",
                        "p": "http://example.org/path_pred",
                        "o": "http://example.org/end",
                        "ot": None,
                    },
                    # Separate path not connected to start or end
                    {
                        "s": "http://example.org/other1",
                        "p": "http://example.org/path_pred",
                        "o": "http://example.org/other2",
                        "ot": None,
                    },
                ],
            )
            conn.commit()

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_translate_mulpath_unbound(self):
        """Test translation of MulPath with both subject and object as variables."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?x ?y
        WHERE {
            ?x ex:path_pred* ?y .
        }"""

        # Translate the query
        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Check that we have the expected projection columns (x and y)
        selected_column_names = {col.key for col in sql_query.selected_columns}
        self.assertEqual(selected_column_names, {"x", "y"})

        # Verify the query is executable
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_translate_mulpath_start_bound(self):
        """Test translation of MulPath with subject bound."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?y
        WHERE {
            ex:start ex:path_pred* ?y .
        }"""

        # Translate the query
        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Check that we have the expected projection column (y)
        selected_column_names = {col.key for col in sql_query.selected_columns}
        self.assertEqual(selected_column_names, {"y"})

        # Verify the query is executable
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_translate_mulpath_end_bound(self):
        """Test translation of MulPath with object bound."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?x
        WHERE {
            ?x ex:path_pred* ex:end .
        }"""

        # Translate the query
        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Check that we have the expected projection column (x)
        selected_column_names = {col.key for col in sql_query.selected_columns}
        self.assertEqual(selected_column_names, {"x"})

        # Verify the query is executable
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_translate_mulpath_both_bound(self):
        """Test translation of MulPath with both endpoints bound."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT *
        WHERE {
            ex:start ex:path_pred* ex:end .
        }"""

        # Translate the query
        sql_query = self.translator.translate(sparql_query)

        # Verify it's a Select statement
        self.assertIsInstance(sql_query, Select)

        # Verify the query is executable
        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_execute_mulpath_unbound(self):
        """Test executing MulPath query with both variables."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?x ?y
        WHERE {
            ?x ex:path_pred* ?y .
        }"""

        # Execute the query
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should find transitive closure of all path_pred edges
        self.assertGreater(len(rows), 0)

        # Convert to set of tuples for easier checking
        result_set = {(row.x, row.y) for row in rows}

        # Should contain the direct edges
        self.assertIn(
            ("http://example.org/start", "http://example.org/node1"), result_set
        )
        self.assertIn(
            ("http://example.org/node1", "http://example.org/node2"), result_set
        )

        # Should contain transitive paths
        self.assertIn(
            ("http://example.org/start", "http://example.org/node2"), result_set
        )
        self.assertIn(
            ("http://example.org/start", "http://example.org/end"), result_set
        )

    def test_execute_mulpath_start_bound(self):
        """Test executing MulPath query with start bound."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?y
        WHERE {
            ex:start ex:path_pred* ?y .
        }"""

        # Execute the query
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should find all nodes reachable from start
        reachable = {row.y for row in rows}

        # Should reach all these nodes
        expected_reachable = {
            "http://example.org/node1",
            "http://example.org/node2",
            "http://example.org/end",
            "http://example.org/alt",
        }

        for node in expected_reachable:
            self.assertIn(node, reachable, f"Should reach {node} from start")

        # Should NOT reach nodes not connected to start
        self.assertNotIn("http://example.org/other1", reachable)
        self.assertNotIn("http://example.org/other2", reachable)

    def test_execute_mulpath_end_bound(self):
        """Test executing MulPath query with end bound."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?x
        WHERE {
            ?x ex:path_pred* ex:end .
        }"""

        # Execute the query
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should find all nodes that can reach end
        can_reach_end = {row.x for row in rows}

        # Should include all these nodes
        expected_sources = {
            "http://example.org/start",
            "http://example.org/node1",
            "http://example.org/node2",
            "http://example.org/alt",
        }

        for node in expected_sources:
            self.assertIn(node, can_reach_end, f"{node} should be able to reach end")

        # Should NOT include nodes that can't reach end
        self.assertNotIn("http://example.org/other1", can_reach_end)
        self.assertNotIn("http://example.org/other2", can_reach_end)

    def test_execute_mulpath_both_bound(self):
        """Test executing MulPath query with both endpoints bound."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        ASK {
            ex:start ex:path_pred* ex:end .
        }"""

        # For ASK queries, we just check if any results exist
        # Since ASK is not implemented, we'll use SELECT instead
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT *
        WHERE {
            ex:start ex:path_pred* ex:end .
        }"""

        # Execute the query
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        # Should find at least one path from start to end
        self.assertGreater(len(rows), 0, "Should find path from start to end")


class TestValueEqualityAndSameTerm(unittest.TestCase):
    """Tests for SPARQL value equality (=) and term identity (sameTerm) semantics.

    Per SPARQL spec:
    - `=` compares values: numeric types are compared by numeric value
    - `sameTerm` compares RDF term identity: both value AND type must match
    """

    def setUp(self):
        """Set up test fixtures with numeric literals of various types."""
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = AlgebraTranslator(
            self.engine, table_name="triples", create_table=True
        )

        # Insert test data with various numeric types
        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    # Same numeric value (1) with different types
                    {
                        "s": "http://ex.org/a",
                        "p": "http://ex.org/val",
                        "o": "1",
                        "ot": str(XSD.integer),
                    },
                    {
                        "s": "http://ex.org/b",
                        "p": "http://ex.org/val",
                        "o": "1.0",
                        "ot": str(XSD.decimal),
                    },
                    {
                        "s": "http://ex.org/c",
                        "p": "http://ex.org/val",
                        "o": "1.0e0",
                        "ot": str(XSD.double),
                    },
                    # Another integer with same value as 'a'
                    {
                        "s": "http://ex.org/d",
                        "p": "http://ex.org/val",
                        "o": "1",
                        "ot": str(XSD.integer),
                    },
                    # Different numeric value
                    {
                        "s": "http://ex.org/e",
                        "p": "http://ex.org/val",
                        "o": "2",
                        "ot": str(XSD.integer),
                    },
                    # String with same lexical form as integer
                    {
                        "s": "http://ex.org/f",
                        "p": "http://ex.org/val",
                        "o": "1",
                        "ot": str(XSD.string),
                    },
                ],
            )
            conn.commit()

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_value_equality_same_numeric_type(self):
        """Test that = matches identical numeric values of same type."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s1 ?s2 WHERE {
            ?s1 ex:val ?v1 .
            ?s2 ex:val ?v2 .
            FILTER(?v1 = ?v2)
            FILTER(?s1 = ex:a)
            FILTER(?s2 = ex:d)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        # a and d both have "1"^^xsd:integer, should match
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].s1, "http://ex.org/a")
        self.assertEqual(rows[0].s2, "http://ex.org/d")

    def test_value_equality_cross_numeric_types(self):
        """Test that = matches numeric values across different numeric types."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s1 ?s2 WHERE {
            ?s1 ex:val ?v1 .
            ?s2 ex:val ?v2 .
            FILTER(?v1 = ?v2)
            FILTER(?s1 < ?s2)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        # Convert to set of pairs for easier checking
        pairs = {(row.s1.split("/")[-1], row.s2.split("/")[-1]) for row in rows}

        # All combinations of a, b, c, d should match (all have numeric value 1)
        expected_numeric_pairs = {
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        }

        for pair in expected_numeric_pairs:
            self.assertIn(
                pair, pairs, f"Numeric values should be equal: {pair[0]} = {pair[1]}"
            )

    def test_value_equality_different_values(self):
        """Test that = does not match different numeric values."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s1 ?s2 WHERE {
            ?s1 ex:val ?v1 .
            ?s2 ex:val ?v2 .
            FILTER(?v1 = ?v2)
            FILTER(?s1 = ex:a)
            FILTER(?s2 = ex:e)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        # a has value 1, e has value 2 - should not match
        self.assertEqual(len(rows), 0)

    def test_value_equality_numeric_vs_string(self):
        """Test that = does not match numeric and string with same lexical form."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s1 ?s2 WHERE {
            ?s1 ex:val ?v1 .
            ?s2 ex:val ?v2 .
            FILTER(?v1 = ?v2)
            FILTER(?s1 = ex:a)
            FILTER(?s2 = ex:f)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        # a is "1"^^xsd:integer, f is "1"^^xsd:string
        # These are incompatible types, should not match
        self.assertEqual(len(rows), 0)

    def test_sameterm_identical_terms(self):
        """Test that sameTerm matches only identical terms (same value AND type)."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s1 ?s2 WHERE {
            ?s1 ex:val ?v1 .
            ?s2 ex:val ?v2 .
            FILTER(sameTerm(?v1, ?v2))
            FILTER(?s1 < ?s2)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        pairs = {(row.s1.split("/")[-1], row.s2.split("/")[-1]) for row in rows}

        # Only a and d should match (both "1"^^xsd:integer)
        self.assertIn(("a", "d"), pairs)

        # Should NOT include pairs with different types even if same numeric value
        self.assertNotIn(("a", "b"), pairs)  # integer vs decimal
        self.assertNotIn(("a", "c"), pairs)  # integer vs double
        self.assertNotIn(("b", "c"), pairs)  # decimal vs double

    def test_sameterm_different_types_same_lexical(self):
        """Test that sameTerm does not match terms with same lexical form but different types."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s1 ?s2 WHERE {
            ?s1 ex:val ?v1 .
            ?s2 ex:val ?v2 .
            FILTER(sameTerm(?v1, ?v2))
            FILTER(?s1 = ex:a)
            FILTER(?s2 = ex:f)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        # a is "1"^^xsd:integer, f is "1"^^xsd:string
        # Same lexical form but different types - sameTerm should NOT match
        self.assertEqual(len(rows), 0)

    def test_value_inequality_cross_types(self):
        """Test that != correctly identifies different values across types."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE {
            ?s ex:val ?v .
            FILTER(?v != "1"^^<http://www.w3.org/2001/XMLSchema#integer>)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        subjects = {row.s.split("/")[-1] for row in rows}

        # e has value 2 (different numeric value) - should be in results
        self.assertIn("e", subjects)

        # f has "1"^^xsd:string (different type) - should be in results
        self.assertIn("f", subjects)

        # a, b, c, d all have numeric value 1 - should NOT be in results
        self.assertNotIn("a", subjects)
        self.assertNotIn("b", subjects)
        self.assertNotIn("c", subjects)
        self.assertNotIn("d", subjects)

    def test_numeric_comparison_less_than(self):
        """Test that < correctly compares numeric values across types."""
        # Insert additional test data
        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://ex.org/g",
                        "p": "http://ex.org/val",
                        "o": "0.5",
                        "ot": str(XSD.decimal),
                    },
                ],
            )
            conn.commit()

        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE {
            ?s ex:val ?v .
            FILTER(?v < "1"^^<http://www.w3.org/2001/XMLSchema#integer>)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        subjects = {row.s.split("/")[-1] for row in rows}

        # g has value 0.5 - should be less than 1
        self.assertIn("g", subjects)

        # a, b, c, d all have value 1 - should NOT be less than 1
        self.assertNotIn("a", subjects)
        self.assertNotIn("b", subjects)
        self.assertNotIn("c", subjects)
        self.assertNotIn("d", subjects)

    def test_numeric_comparison_greater_than(self):
        """Test that > correctly compares numeric values across types."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE {
            ?s ex:val ?v .
            FILTER(?v > "1"^^<http://www.w3.org/2001/XMLSchema#integer>)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        subjects = {row.s.split("/")[-1] for row in rows}

        # e has value 2 - should be greater than 1
        self.assertIn("e", subjects)

        # a, b, c, d all have value 1 - should NOT be greater than 1
        self.assertNotIn("a", subjects)
        self.assertNotIn("b", subjects)
        self.assertNotIn("c", subjects)
        self.assertNotIn("d", subjects)


class TestValueEqualityWithLiterals(unittest.TestCase):
    """Tests for value equality with literal values in FILTER expressions."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = AlgebraTranslator(
            self.engine, table_name="triples", create_table=True
        )

        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://ex.org/x",
                        "p": "http://ex.org/num",
                        "o": "42",
                        "ot": str(XSD.integer),
                    },
                    {
                        "s": "http://ex.org/y",
                        "p": "http://ex.org/num",
                        "o": "42.0",
                        "ot": str(XSD.decimal),
                    },
                    {
                        "s": "http://ex.org/z",
                        "p": "http://ex.org/num",
                        "o": "4.2e1",
                        "ot": str(XSD.double),
                    },
                ],
            )
            conn.commit()

    def tearDown(self):
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_filter_equality_with_integer_literal(self):
        """Test FILTER(?v = 42) matches all numeric representations of 42."""
        query = """
        PREFIX ex: <http://ex.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?s WHERE {
            ?s ex:num ?v .
            FILTER(?v = "42"^^xsd:integer)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        subjects = {row.s.split("/")[-1] for row in rows}

        # All three should match (42 as integer, 42.0 as decimal, 4.2e1 as double)
        self.assertEqual(subjects, {"x", "y", "z"})

    def test_filter_equality_with_decimal_literal(self):
        """Test FILTER(?v = 42.0) matches all numeric representations of 42."""
        query = """
        PREFIX ex: <http://ex.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?s WHERE {
            ?s ex:num ?v .
            FILTER(?v = "42.0"^^xsd:decimal)
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        subjects = {row.s.split("/")[-1] for row in rows}

        # All three should match
        self.assertEqual(subjects, {"x", "y", "z"})


if __name__ == "__main__":
    unittest.main()
