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


if __name__ == "__main__":
    unittest.main()
