"""Tests for the sparql2sql module.

Most testing is performed by the W3C tests, but these are
small sanity checks of the implementation.
"""

import unittest
from rdflib import XSD
from sqlalchemy import create_engine
from sqlalchemy.sql.selectable import Select, CTE

from sparql2sql.sparql2sql import Translator, create_databricks_engine


class TestTranslator(unittest.TestCase):
    """Tests for the Translator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = Translator(
            self.engine, table_name="triples", create_table=True
        )

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_translate_path_expression_query(self):
        """Test translation of a SPARQL query with path expressions."""
        sparql_query = """
        PREFIX dbx: <http://www.databricks.com/ontology/>
        SELECT *
        WHERE {
            ?cat a dbx:Catalog .
            ?tab dbx:tableInSchema/dbx:inCatalog ?cat .
        }"""

        _, sql_query = self.translator.translate(sparql_query)
        self.assertIsInstance(sql_query, Select)

        selected_column_names = {col.key for col in sql_query.selected_columns}
        self.assertEqual(selected_column_names, {"cat", "tab"})

        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_translate_optional_query(self):
        """Test translation of a SPARQL query with OPTIONAL (LEFT JOIN)."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT *
        WHERE {
            ?person ex:name ?name .
            OPTIONAL { ?person ex:age ?age }
        }"""

        _, sql_query = self.translator.translate(sparql_query)
        self.assertIsInstance(sql_query, Select)

        selected_column_names = {col.key for col in sql_query.selected_columns}
        self.assertEqual(selected_column_names, {"person", "name", "age"})

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

        _, sql_query = self.translator.translate(sparql_query)
        self.assertIsInstance(sql_query, Select)
        self.assertIsNotNone(sql_query._order_by_clauses)
        self.assertEqual(len(sql_query._order_by_clauses), 1)

        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
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

        _, sql_query = self.translator.translate(sparql_query)
        self.assertIsInstance(sql_query, Select)
        self.assertIsNotNone(sql_query._order_by_clauses)
        self.assertEqual(len(sql_query._order_by_clauses), 1)

        try:
            compiled = sql_query.compile()
            self.assertIsNotNone(compiled)
            sql_str = str(compiled)
            self.assertIn("ORDER BY", sql_str)
            self.assertIn("DESC", sql_str)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")


class TestTranslatorExecute(unittest.TestCase):
    """Tests for executing queries against sample data."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = Translator(
            self.engine, table_name="triples", create_table=True
        )

        with self.engine.connect() as conn:
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
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_execute_path_expression_query(self):
        """Test executing a SPARQL query with path expressions."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT *
        WHERE {
            ?person ex:parent/ex:parent ?grandparent .
        }"""

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.person, "http://example.org/alice")
        self.assertEqual(row.grandparent, "http://example.org/charlie")

    def test_execute_optional_query(self):
        """Test executing a SPARQL query with OPTIONAL."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT *
        WHERE {
            ?person ex:name ?name .
            OPTIONAL { ?person ex:age ?age }
        }"""

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        self.assertEqual(len(rows), 3)
        results_dict = {row.person: {"name": row.name, "age": row.age} for row in rows}

        self.assertIn("http://example.org/alice", results_dict)
        self.assertEqual(results_dict["http://example.org/alice"]["name"], "Alice")
        self.assertIsNone(results_dict["http://example.org/alice"]["age"])

        self.assertIn("http://example.org/bob", results_dict)
        self.assertEqual(results_dict["http://example.org/bob"]["name"], "Bob")
        self.assertEqual(results_dict["http://example.org/bob"]["age"], "30")

        self.assertIn("http://example.org/charlie", results_dict)
        self.assertEqual(results_dict["http://example.org/charlie"]["name"], "Charlie")
        self.assertIsNone(results_dict["http://example.org/charlie"]["age"])

    def test_execute_same_variable_triple_pattern(self):
        """Test executing a query where the same variable appears multiple times."""
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

        sparql_query = "SELECT ?x WHERE { ?x ?x ?x }"
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].x, "http://example.org/same")

    def test_execute_empty_bgp(self):
        """Test executing a query with an empty BGP."""
        sparql_query = "SELECT * WHERE {}"
        result = self.translator.execute(sparql_query)
        rows = result.fetchall()
        self.assertEqual(len(rows), 1)

    def test_execute_distinct(self):
        """Test that SELECT DISTINCT removes duplicate rows."""
        result = self.translator.execute("SELECT ?person WHERE { ?person ?p ?o }")
        rows = result.fetchall()
        persons = [row.person for row in rows]
        self.assertGreater(
            len(persons), len(set(persons)), "Should have duplicates without DISTINCT"
        )

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
        self.assertEqual(names, ["Charlie", "Bob", "Alice"])

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
        self.assertEqual(len(rows), 2)
        self.assertEqual(names, ["Bob", "Charlie"])


class TestMulPath(unittest.TestCase):
    """Tests for MulPath translation and execution."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = Translator(
            self.engine, table_name="triples", create_table=True
        )

        with self.engine.connect() as conn:
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
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_execute_mulpath_unbound(self):
        """Test executing MulPath query with both variables."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?x ?y
        WHERE {
            ?x ex:path_pred* ?y .
        }"""

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        self.assertGreater(len(rows), 0)
        result_set = {(row.x, row.y) for row in rows}

        self.assertIn(
            ("http://example.org/start", "http://example.org/node1"), result_set
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

        result = self.translator.execute(sparql_query)
        rows = result.fetchall()

        reachable = {row.y for row in rows}
        expected = {
            "http://example.org/node1",
            "http://example.org/node2",
            "http://example.org/end",
            "http://example.org/alt",
        }

        for node in expected:
            self.assertIn(node, reachable)

        self.assertNotIn("http://example.org/other1", reachable)

    def test_zero_length_star_includes_start(self):
        """Test p* includes zero-length path (start node itself)."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?y WHERE { ex:start ex:path_pred* ?y }
        """
        result = self.translator.execute(sparql_query)
        reachable = {row.y for row in result.fetchall()}

        # Zero-length path means start connects to itself
        self.assertIn("http://example.org/start", reachable)

    def test_zero_length_plus_excludes_start(self):
        """Test p+ excludes zero-length path (one-or-more only)."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?y WHERE { ex:start ex:path_pred+ ?y }
        """
        result = self.translator.execute(sparql_query)
        reachable = {row.y for row in result.fetchall()}

        # p+ requires at least one hop - start should NOT be included
        self.assertNotIn("http://example.org/start", reachable)
        # But reachable nodes should still be found
        self.assertIn("http://example.org/node1", reachable)
        self.assertIn("http://example.org/end", reachable)

    def test_zero_or_one_includes_start_and_one_hop(self):
        """Test p? includes zero-length and exactly one hop."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?y WHERE { ex:start ex:path_pred? ?y }
        """
        result = self.translator.execute(sparql_query)
        reachable = {row.y for row in result.fetchall()}

        # Zero-length: start itself
        self.assertIn("http://example.org/start", reachable)
        # One hop: node1 and alt (direct neighbours)
        self.assertIn("http://example.org/node1", reachable)
        self.assertIn("http://example.org/alt", reachable)
        # Two hops: should NOT be included
        self.assertNotIn("http://example.org/node2", reachable)
        self.assertNotIn("http://example.org/end", reachable)

    def test_zero_length_with_both_variables(self):
        """Test p* with unbound variables includes all self-loops."""
        sparql_query = """
        PREFIX ex: <http://example.org/>
        SELECT ?x ?y WHERE { ?x ex:path_pred* ?y }
        """
        result = self.translator.execute(sparql_query)
        pairs = {(row.x, row.y) for row in result.fetchall()}

        # All graph nodes should have zero-length self-loops
        graph_nodes = {
            "http://example.org/start",
            "http://example.org/node1",
            "http://example.org/node2",
            "http://example.org/end",
            "http://example.org/alt",
            "http://example.org/other1",
            "http://example.org/other2",
        }
        for node in graph_nodes:
            self.assertIn((node, node), pairs, f"Missing zero-length for {node}")


class TestNestedPaths(unittest.TestCase):
    """Tests for nested path expressions (e.g., p/q*, p*/q)."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = Translator(
            self.engine, table_name="triples", create_table=True
        )

        # Graph structure:
        # alice --knows--> bob --knows--> charlie --knows--> diana
        #                      \--likes--> eve
        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://example.org/alice",
                        "p": "http://example.org/knows",
                        "o": "http://example.org/bob",
                        "ot": None,
                    },
                    {
                        "s": "http://example.org/bob",
                        "p": "http://example.org/knows",
                        "o": "http://example.org/charlie",
                        "ot": None,
                    },
                    {
                        "s": "http://example.org/charlie",
                        "p": "http://example.org/knows",
                        "o": "http://example.org/diana",
                        "ot": None,
                    },
                    {
                        "s": "http://example.org/bob",
                        "p": "http://example.org/likes",
                        "o": "http://example.org/eve",
                        "ot": None,
                    },
                ],
            )
            conn.commit()

    def tearDown(self):
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_sequence_then_mulpath(self):
        """Test p/q* - sequence followed by transitive closure."""
        # alice --knows--> bob --knows*--> {charlie, diana}
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?end
        WHERE { ex:alice ex:knows/ex:knows* ?end }
        """
        result = self.translator.execute(query)
        ends = {row.end for row in result.fetchall()}

        # bob --knows*--> bob (via 0+ hops), charlie (1 hop), diana (2 hops)
        # Note: zero-length path not yet implemented, so bob may not appear
        self.assertIn("http://example.org/charlie", ends)
        self.assertIn("http://example.org/diana", ends)

    def test_mulpath_then_sequence(self):
        """Test p*/q - transitive closure followed by simple predicate."""
        # alice --knows*--> {alice, bob, charlie, diana} --likes--> eve
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?end
        WHERE { ex:alice ex:knows*/ex:likes ?end }
        """
        result = self.translator.execute(query)
        ends = {row.end for row in result.fetchall()}

        # Only bob has a likes edge, so result should be eve
        self.assertIn("http://example.org/eve", ends)

    def test_mulpath_in_middle_of_sequence(self):
        """Test p/q*/r - simple, transitive, simple."""
        # alice --knows--> bob --knows*--> X --likes--> eve
        # With zero-length path support, bob matches knows* with 0 hops
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?end
        WHERE { ex:alice ex:knows/ex:knows*/ex:likes ?end }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        # Should find path: alice -> bob -> (0 hops via knows*) -> likes -> eve
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].end, "http://example.org/eve")

    @unittest.expectedFailure  # TODO: CTE name collision with multiple MulPaths
    def test_two_mulpaths_in_sequence(self):
        """Test p*/q* - two transitive closures in sequence."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?start ?end
        WHERE { ?start ex:knows*/ex:likes* ?end }
        """
        _, sql_query = self.translator.translate(query)

        # Should compile without error
        compiled = sql_query.compile()
        self.assertIsNotNone(compiled)

    def test_nested_path_translation_compiles(self):
        """Test that nested paths produce valid SQL."""
        queries = [
            "SELECT * WHERE { ?s <p>/<q>* ?o }",
            "SELECT * WHERE { ?s <p>*/<q> ?o }",
            "SELECT * WHERE { ?s <p>/<q>+/<r> ?o }",
            "SELECT * WHERE { ?s ^<p>/<q> ?o }",
            "SELECT * WHERE { ?s <p>/^<q> ?o }",
        ]

        for sparql in queries:
            with self.subTest(sparql=sparql):
                _, sql_query = self.translator.translate(sparql)
                compiled = sql_query.compile()
                self.assertIsNotNone(compiled)

    def test_multiple_mulpaths_compile(self):
        """Test that multiple MulPaths in sequence compile correctly."""
        query = "SELECT * WHERE { ?s <p>?/<q>* ?o }"
        _, sql_query = self.translator.translate(query)
        compiled = sql_query.compile()
        self.assertIsNotNone(compiled)

    def test_invpath_simple(self):
        """Test simple inverse path (^) swaps subject and object."""
        # Normal: alice --knows--> bob
        # Inverse: bob <--knows-- alice => bob ^knows alice
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?who
        WHERE { ex:bob ^ex:knows ?who }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        # Should find alice (who knows bob)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].who, "http://example.org/alice")

    def test_invpath_in_sequence(self):
        """Test inverse path in sequence: ^p/q."""
        # Query: ?x ^knows/likes ?y
        # Expands to: ?temp knows ?x, ?temp likes ?y
        # bob knows charlie, bob likes eve => (charlie, eve)
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?x ?y
        WHERE { ?x ^ex:knows/ex:likes ?y }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        # bob knows charlie and bob likes eve => (charlie, eve)
        found = {(row.x, row.y) for row in rows}
        self.assertIn(("http://example.org/charlie", "http://example.org/eve"), found)

    @unittest.expectedFailure  # TODO: complex inner paths in MulPath not yet supported
    def test_invpath_nested_in_mulpath(self):
        """Test inverse path with transitive closure: (^p)*."""
        # alice --knows--> bob
        # Using (^knows)* from bob should reach alice (1 hop) and bob (0 hops)
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?who
        WHERE { ex:bob (^ex:knows)* ?who }
        """
        result = self.translator.execute(query)
        whos = {row.who for row in result.fetchall()}

        # Should include bob (zero-length) and alice (one inverse hop)
        self.assertIn("http://example.org/bob", whos)
        self.assertIn("http://example.org/alice", whos)

    def test_negated_path_simple(self):
        """Test simple negated path (!p) excludes specified predicate."""
        # Graph: alice --knows--> bob, bob --knows--> charlie, bob --likes--> eve
        # Query: bob !knows ?o => should get eve (via likes), not charlie (via knows)
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?o
        WHERE { ex:bob !ex:knows ?o }
        """
        result = self.translator.execute(query)
        objs = {row.o for row in result.fetchall()}

        self.assertIn("http://example.org/eve", objs)
        self.assertNotIn("http://example.org/charlie", objs)

    def test_negated_path_multiple(self):
        """Test negated path with multiple exclusions !(p|q)."""
        # Exclude both knows and likes
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?s ?o
        WHERE { ?s !(ex:knows|ex:likes) ?o }
        """
        result = self.translator.execute(query)
        preds_found = {(row.s, row.o) for row in result.fetchall()}

        # Should NOT find any knows or likes edges
        self.assertNotIn(
            ("http://example.org/alice", "http://example.org/bob"), preds_found
        )
        self.assertNotIn(
            ("http://example.org/bob", "http://example.org/eve"), preds_found
        )

    def test_negated_inverse_path(self):
        """Test negated inverse path (!^p)."""
        # Query: bob !^knows ?who => find ?who where NOT (?who knows bob)
        # alice knows bob, so alice should be excluded
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?who
        WHERE { ex:bob !^ex:knows ?who }
        """
        result = self.translator.execute(query)
        whos = {row.who for row in result.fetchall()}

        # alice knows bob, so she should NOT appear
        self.assertNotIn("http://example.org/alice", whos)

    def test_negated_path_compiles(self):
        """Test that negated paths produce valid SQL."""
        queries = [
            "SELECT * WHERE { ?s !<p> ?o }",
            "SELECT * WHERE { ?s !(<p>|<q>) ?o }",
            "SELECT * WHERE { ?s !^<p> ?o }",
            "SELECT * WHERE { ?s !(<p>|^<q>) ?o }",
        ]
        for sparql in queries:
            with self.subTest(sparql=sparql):
                _, sql_query = self.translator.translate(sparql)
                compiled = sql_query.compile()
                self.assertIsNotNone(compiled)


class TestGroupByAndAggregates(unittest.TestCase):
    """Tests for GROUP BY and aggregate functions."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = Translator(
            self.engine, table_name="triples", create_table=True
        )

        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    {
                        "s": "http://ex.org/alice",
                        "p": "http://ex.org/dept",
                        "o": "Engineering",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://ex.org/alice",
                        "p": "http://ex.org/salary",
                        "o": "100",
                        "ot": str(XSD.integer),
                    },
                    {
                        "s": "http://ex.org/bob",
                        "p": "http://ex.org/dept",
                        "o": "Engineering",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://ex.org/bob",
                        "p": "http://ex.org/salary",
                        "o": "120",
                        "ot": str(XSD.integer),
                    },
                    {
                        "s": "http://ex.org/charlie",
                        "p": "http://ex.org/dept",
                        "o": "Sales",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://ex.org/charlie",
                        "p": "http://ex.org/salary",
                        "o": "80",
                        "ot": str(XSD.integer),
                    },
                    {
                        "s": "http://ex.org/diana",
                        "p": "http://ex.org/dept",
                        "o": "Sales",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://ex.org/diana",
                        "p": "http://ex.org/salary",
                        "o": "90",
                        "ot": str(XSD.integer),
                    },
                ],
            )
            conn.commit()

    def tearDown(self):
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_execute_group_by_count(self):
        """Test GROUP BY with COUNT returns correct counts per group."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?dept (COUNT(?person) AS ?count)
        WHERE { ?person ex:dept ?dept }
        GROUP BY ?dept
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        counts = {row.dept: int(row.count) for row in rows}

        self.assertEqual(counts["Engineering"], 2)
        self.assertEqual(counts["Sales"], 2)

    def test_execute_group_by_sum(self):
        """Test GROUP BY with SUM returns correct totals per group."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?dept (SUM(?salary) AS ?total)
        WHERE {
            ?person ex:dept ?dept .
            ?person ex:salary ?salary .
        }
        GROUP BY ?dept
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        totals = {row.dept: float(row.total) for row in rows}

        self.assertEqual(totals["Engineering"], 220.0)
        self.assertEqual(totals["Sales"], 170.0)

    def test_execute_aggregate_without_group_by(self):
        """Test aggregate over entire result set (no GROUP BY)."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT (COUNT(?person) AS ?total)
        WHERE { ?person ex:dept ?dept }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0].total), 4)


class TestGraphPatterns(unittest.TestCase):
    """Tests for GRAPH pattern support."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = Translator(
            self.engine, table_name="triples", create_table=True, graph_aware=True
        )

        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    # Default graph
                    {
                        "s": "http://example.org/s1",
                        "p": "http://example.org/p1",
                        "o": "default_obj",
                        "ot": str(XSD.string),
                        "g": None,
                    },
                    # Named graph 1
                    {
                        "s": "http://example.org/s2",
                        "p": "http://example.org/p2",
                        "o": "graph1_obj",
                        "ot": str(XSD.string),
                        "g": "http://example.org/graph1",
                    },
                    # Named graph 2
                    {
                        "s": "http://example.org/s3",
                        "p": "http://example.org/p3",
                        "o": "graph2_obj",
                        "ot": str(XSD.string),
                        "g": "http://example.org/graph2",
                    },
                ],
            )
            conn.commit()

    def tearDown(self):
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_query_default_graph(self):
        """Test querying the default graph."""
        query = "SELECT ?s ?o WHERE { ?s ?p ?o }"
        result = self.translator.execute(query)
        rows = result.fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].o, "default_obj")

    def test_query_named_graph(self):
        """Test querying a specific named graph."""
        query = """
        SELECT ?s ?o WHERE {
            GRAPH <http://example.org/graph1> { ?s ?p ?o }
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].o, "graph1_obj")

    def test_query_graph_variable(self):
        """Test querying with a graph variable."""
        query = """
        SELECT ?g ?o WHERE {
            GRAPH ?g { ?s ?p ?o }
        }
        """
        result = self.translator.execute(query)
        rows = result.fetchall()

        self.assertEqual(len(rows), 2)
        graphs = {row.g for row in rows}
        self.assertEqual(
            graphs, {"http://example.org/graph1", "http://example.org/graph2"}
        )


class TestEffectiveBooleanValue(unittest.TestCase):
    """Tests for Effective Boolean Value (EBV) semantics per SPARQL 17.2.2."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.translator = Translator(
            self.engine, table_name="triples", create_table=True
        )

        with self.engine.connect() as conn:
            conn.execute(
                self.translator.table.insert(),
                [
                    # Boolean values
                    {
                        "s": "http://ex.org/b1",
                        "p": "http://ex.org/val",
                        "o": "true",
                        "ot": str(XSD.boolean),
                    },
                    {
                        "s": "http://ex.org/b2",
                        "p": "http://ex.org/val",
                        "o": "false",
                        "ot": str(XSD.boolean),
                    },
                    # String values
                    {
                        "s": "http://ex.org/s1",
                        "p": "http://ex.org/val",
                        "o": "hello",
                        "ot": str(XSD.string),
                    },
                    {
                        "s": "http://ex.org/s2",
                        "p": "http://ex.org/val",
                        "o": "",
                        "ot": str(XSD.string),
                    },
                    # Plain literals (no datatype)
                    {
                        "s": "http://ex.org/p1",
                        "p": "http://ex.org/val",
                        "o": "plain",
                        "ot": None,
                    },
                    {
                        "s": "http://ex.org/p2",
                        "p": "http://ex.org/val",
                        "o": "",
                        "ot": None,
                    },
                    # Numeric values
                    {
                        "s": "http://ex.org/n1",
                        "p": "http://ex.org/val",
                        "o": "42",
                        "ot": str(XSD.integer),
                    },
                    {
                        "s": "http://ex.org/n2",
                        "p": "http://ex.org/val",
                        "o": "0",
                        "ot": str(XSD.integer),
                    },
                    {
                        "s": "http://ex.org/n3",
                        "p": "http://ex.org/val",
                        "o": "3.14",
                        "ot": str(XSD.decimal),
                    },
                    {
                        "s": "http://ex.org/n4",
                        "p": "http://ex.org/val",
                        "o": "0.0",
                        "ot": str(XSD.decimal),
                    },
                    # Unknown datatype (should be type error)
                    {
                        "s": "http://ex.org/u1",
                        "p": "http://ex.org/val",
                        "o": "2024-01-01",
                        "ot": str(XSD.date),
                    },
                ],
            )
            conn.commit()

    def tearDown(self):
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_ebv_boolean_true(self):
        """EBV of xsd:boolean 'true' is true."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        self.assertIn("http://ex.org/b1", subjects)

    def test_ebv_boolean_false(self):
        """EBV of xsd:boolean 'false' is false."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        self.assertNotIn("http://ex.org/b2", subjects)

    def test_ebv_string_nonempty(self):
        """EBV of non-empty xsd:string is true."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        self.assertIn("http://ex.org/s1", subjects)

    def test_ebv_string_empty(self):
        """EBV of empty xsd:string is false."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        self.assertNotIn("http://ex.org/s2", subjects)

    def test_ebv_plain_literal_nonempty(self):
        """EBV of non-empty plain literal is true."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        self.assertIn("http://ex.org/p1", subjects)

    def test_ebv_plain_literal_empty(self):
        """EBV of empty plain literal is false."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        self.assertNotIn("http://ex.org/p2", subjects)

    def test_ebv_numeric_nonzero(self):
        """EBV of non-zero numeric is true."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        self.assertIn("http://ex.org/n1", subjects)
        self.assertIn("http://ex.org/n3", subjects)

    def test_ebv_numeric_zero(self):
        """EBV of zero numeric is false."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        self.assertNotIn("http://ex.org/n2", subjects)
        self.assertNotIn("http://ex.org/n4", subjects)

    def test_ebv_unknown_type_error(self):
        """EBV of unknown datatype is type error (filters out row)."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        # xsd:date is not a valid EBV type, so u1 should be filtered out
        self.assertNotIn("http://ex.org/u1", subjects)

    def test_ebv_not_operator(self):
        """NOT applies EBV then negates."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE { ?s ex:val ?v . FILTER(!?v) }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        # NOT(true EBV) = false, NOT(false EBV) = true
        self.assertIn("http://ex.org/b2", subjects)  # false boolean
        self.assertIn("http://ex.org/s2", subjects)  # empty string
        self.assertIn("http://ex.org/n2", subjects)  # zero integer
        self.assertNotIn("http://ex.org/b1", subjects)  # true boolean
        self.assertNotIn("http://ex.org/s1", subjects)  # non-empty string

    def test_ebv_or_operator(self):
        """OR applies EBV to operands."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE {
            ?s ex:val ?v .
            FILTER(?v || false)
        }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        # true OR false = true for truthy values
        self.assertIn("http://ex.org/b1", subjects)
        self.assertIn("http://ex.org/s1", subjects)
        self.assertIn("http://ex.org/n1", subjects)
        # false OR false = false for falsy values
        self.assertNotIn("http://ex.org/b2", subjects)
        self.assertNotIn("http://ex.org/s2", subjects)

    def test_ebv_and_operator(self):
        """AND applies EBV to operands."""
        query = """
        PREFIX ex: <http://ex.org/>
        SELECT ?s WHERE {
            ?s ex:val ?v .
            FILTER(?v && true)
        }
        """
        result = self.translator.execute(query)
        subjects = {row.s for row in result.fetchall()}
        # true AND true = true for truthy values
        self.assertIn("http://ex.org/b1", subjects)
        self.assertIn("http://ex.org/s1", subjects)
        # false AND true = false for falsy values
        self.assertNotIn("http://ex.org/b2", subjects)
        self.assertNotIn("http://ex.org/s2", subjects)


class TestBnodeIsomorphism(unittest.TestCase):
    """Tests for the blank node isomorphism checker used in W3C test comparison.

    This is a 'test of tests' - ensuring the test infrastructure correctly
    identifies isomorphic and non-isomorphic result sets with blank nodes.
    """

    def setUp(self):
        from tests.w3c_test_base import _bnode_isomorphic

        self.isomorphic = _bnode_isomorphic

    # === ISOMORPHIC CASES (should return True) ===

    def test_identical_no_bnodes(self):
        """Identical rows without bnodes are isomorphic."""
        self.assertTrue(
            self.isomorphic(
                [("alice", "knows", "bob")],
                [("alice", "knows", "bob")],
            )
        )

    def test_same_structure_different_bnode_ids(self):
        """Same structure with different bnode IDs is isomorphic."""
        self.assertTrue(
            self.isomorphic(
                [("_:a1", "knows", "_:b1"), ("_:b1", "knows", "_:a1")],
                [("_:x9", "knows", "_:y9"), ("_:y9", "knows", "_:x9")],
            )
        )

    def test_different_order(self):
        """Row order doesn't matter (multiset semantics)."""
        self.assertTrue(
            self.isomorphic(
                [("_:a", "knows", "_:b"), ("_:c", "likes", "_:d")],
                [("_:z", "likes", "_:w"), ("_:x", "knows", "_:y")],
            )
        )

    def test_duplicate_rows(self):
        """Duplicate rows must match in count."""
        self.assertTrue(
            self.isomorphic(
                [("_:a", "type", "Person"), ("_:a", "type", "Person")],
                [("_:x", "type", "Person"), ("_:x", "type", "Person")],
            )
        )

    def test_bnode_chain(self):
        """Chain of bnodes is isomorphic."""
        self.assertTrue(
            self.isomorphic(
                [("_:a", "next", "_:b"), ("_:b", "next", "_:c")],
                [("_:1", "next", "_:2"), ("_:2", "next", "_:3")],
            )
        )

    def test_self_referential_bnode(self):
        """Self-referential bnode is isomorphic."""
        self.assertTrue(
            self.isomorphic(
                [("_:a", "knows", "_:a")],
                [("_:x", "knows", "_:x")],
            )
        )

    def test_empty_result_sets(self):
        """Empty result sets are isomorphic."""
        self.assertTrue(self.isomorphic([], []))

    # === NON-ISOMORPHIC CASES (should return False) ===

    def test_different_literal_values(self):
        """Different literal values are not isomorphic."""
        self.assertFalse(
            self.isomorphic(
                [("_:a", "name", "Alice")],
                [("_:x", "name", "Bob")],
            )
        )

    def test_different_row_counts(self):
        """Different row counts are not isomorphic."""
        self.assertFalse(
            self.isomorphic(
                [("_:a", "knows", "_:b"), ("_:b", "knows", "_:a")],
                [("_:x", "knows", "_:y")],
            )
        )

    def test_bnode_vs_uri(self):
        """Bnode cannot map to URI."""
        self.assertFalse(
            self.isomorphic(
                [("_:a", "type", "Person")],
                [("http://alice", "type", "Person")],
            )
        )

    def test_non_bijective_two_to_one(self):
        """Two distinct bnodes cannot map to one (non-bijective)."""
        self.assertFalse(
            self.isomorphic(
                [("_:a", "knows", "_:b")],  # 2 distinct bnodes
                [("_:x", "knows", "_:x")],  # 1 bnode (self-ref)
            )
        )

    def test_different_bnode_connections(self):
        """Different connection structure is not isomorphic."""
        self.assertFalse(
            self.isomorphic(
                [("_:a", "knows", "_:b"), ("_:a", "knows", "_:c")],  # a -> b, a -> c
                [("_:x", "knows", "_:y"), ("_:y", "knows", "_:z")],  # x -> y -> z
            )
        )

    def test_inconsistent_bnode_mapping(self):
        """Bnode mapping must be consistent across rows."""
        self.assertFalse(
            self.isomorphic(
                [("_:a", "r1", "_:b"), ("_:a", "r2", "_:c")],  # same _:a in both
                [("_:x", "r1", "_:y"), ("_:z", "r2", "_:w")],  # different x vs z
            )
        )

    # === EDGE CASES ===

    def test_complex_valid_isomorphism(self):
        """Complex 4-bnode valid isomorphism."""
        self.assertTrue(
            self.isomorphic(
                [("_:a", "r", "_:b"), ("_:c", "r", "_:d"), ("_:a", "s", "_:c")],
                [("_:1", "r", "_:2"), ("_:3", "r", "_:4"), ("_:1", "s", "_:3")],
            )
        )

    def test_complex_invalid_isomorphism(self):
        """Complex 4-bnode invalid isomorphism (wrong correspondence)."""
        self.assertFalse(
            self.isomorphic(
                [("_:a", "r", "_:b"), ("_:c", "r", "_:d"), ("_:a", "s", "_:c")],
                [
                    ("_:1", "r", "_:2"),
                    ("_:3", "r", "_:4"),
                    ("_:1", "s", "_:4"),
                ],  # 4 not 3
            )
        )


if __name__ == "__main__":
    unittest.main()
