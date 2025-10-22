import unittest
from unittest.mock import Mock, patch, MagicMock
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
            self.engine, 
            table_name="triples", 
            create_table=True
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Dispose of the engine to clean up resources
        if hasattr(self, 'engine'):
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
        self.assertEqual(selected_column_names, {'cat', 'tab'})
        
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
        self.assertEqual(selected_column_names, {'person', 'name', 'age'})
        
        # The query should select from a CTE (since LEFT JOIN creates a CTE)
        froms = sql_query.get_final_froms()
        self.assertEqual(len(froms), 1)
        
        # The FROM should be a CTE
        cte = froms[0]
        self.assertIsInstance(cte, CTE)
        
        # The CTE should be a SELECT with a LEFT JOIN
        # We can check this by looking at the CTE's original element
        cte_select = cte.original
        self.assertIsInstance(cte_select, Select)
        
        # Verify there's a FROM clause (which should contain the LEFT JOIN)
        cte_froms = cte_select.get_final_froms()
        self.assertGreater(len(cte_froms), 0)
        
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
            self.engine, 
            table_name="triples", 
            create_table=True
        )
        
        # Insert sample data
        with self.engine.connect() as conn:
            # Data for path expression tests
            # Relationships: alice -> bob -> charlie (parent chain)
            conn.execute(self.translator.table.insert(), [
                {'s': '<http://example.org/alice>', 'p': '<http://example.org/parent>', 'o': '<http://example.org/bob>'},
                {'s': '<http://example.org/bob>', 'p': '<http://example.org/parent>', 'o': '<http://example.org/charlie>'},
            ])
            
            # Data for OPTIONAL tests
            # alice has name "Alice"
            # bob has name "Bob" and age "30"
            # charlie has name "Charlie" (no age)
            conn.execute(self.translator.table.insert(), [
                {'s': '<http://example.org/alice>', 'p': '<http://example.org/name>', 'o': 'Alice'},
                {'s': '<http://example.org/bob>', 'p': '<http://example.org/name>', 'o': 'Bob'},
                {'s': '<http://example.org/bob>', 'p': '<http://example.org/age>', 'o': '30'},
                {'s': '<http://example.org/charlie>', 'p': '<http://example.org/name>', 'o': 'Charlie'},
            ])
            conn.commit()

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, 'engine'):
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
        self.assertEqual(row.person, '<http://example.org/alice>')
        self.assertEqual(row.grandparent, '<http://example.org/charlie>')

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
        results_dict = {row.person: {'name': row.name, 'age': row.age} for row in rows}
        
        # Verify alice (has name, no age)
        self.assertIn('<http://example.org/alice>', results_dict)
        self.assertEqual(results_dict['<http://example.org/alice>']['name'], 'Alice')
        self.assertIsNone(results_dict['<http://example.org/alice>']['age'])
        
        # Verify bob (has both name and age)
        self.assertIn('<http://example.org/bob>', results_dict)
        self.assertEqual(results_dict['<http://example.org/bob>']['name'], 'Bob')
        self.assertEqual(results_dict['<http://example.org/bob>']['age'], '30')
        
        # Verify charlie (has name, no age)
        self.assertIn('<http://example.org/charlie>', results_dict)
        self.assertEqual(results_dict['<http://example.org/charlie>']['name'], 'Charlie')
        self.assertIsNone(results_dict['<http://example.org/charlie>']['age'])


if __name__ == '__main__':
    unittest.main()

