"""
W3C SPARQL Evaluation Test Suite for sqlalch.py

This module runs the W3C SPARQL 1.0 evaluation test cases against the
AlgebraTranslator implementation to verify conformance with the SPARQL spec.

Test suite source: https://www.w3.org/2001/sw/DataAccess/tests/data-r2/
Manifest: https://www.w3.org/2001/sw/DataAccess/tests/data-r2/manifest-evaluation.ttl

We only run evaluation tests (not syntax tests since rdflib handles parsing)
and filter for SELECT queries since that's what's currently implemented.
"""

import os
import ssl
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS
from sqlalchemy import create_engine

from rdfdf.sqlalch import AlgebraTranslator, term_to_string


# W3C Test Suite namespaces
MF = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#")
QT = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-query#")
RS = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/result-set#")

# Base URL for the test suite
W3C_TEST_BASE = "https://www.w3.org/2001/sw/DataAccess/tests/data-r2/"
MANIFEST_URL = W3C_TEST_BASE + "manifest-evaluation.ttl"

# Local cache directory for downloaded test files
CACHE_DIR = Path(__file__).parent / ".w3c_test_cache"


def ensure_cache_dir():
    """Create the cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, force: bool = False) -> str:
    """Download a file from URL and cache it locally.

    Args:
        url: The URL to download
        force: If True, re-download even if cached

    Returns:
        The local file path
    """
    ensure_cache_dir()

    # Create a safe filename from the URL
    # Keep the path structure for organization
    relative_path = url.replace(W3C_TEST_BASE, "")
    if relative_path.startswith("http"):
        # URL is not under the test base, use a hash
        relative_path = (
            url.replace("https://", "").replace("http://", "").replace("/", "_")
        )

    local_path = CACHE_DIR / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and not force:
        return str(local_path)

    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(url, timeout=30, context=ssl_context) as response:
            content = response.read()
            local_path.write_bytes(content)
        return str(local_path)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def load_graph(url_or_path: str) -> Graph:
    """Load an RDF graph from a URL or local path.

    Args:
        url_or_path: URL or local file path

    Returns:
        Parsed RDF Graph
    """
    g = Graph()

    if url_or_path.startswith("http"):
        local_path = download_file(url_or_path)
    else:
        local_path = url_or_path

    # Determine format from file extension
    ext = Path(local_path).suffix.lower()
    format_map = {
        ".ttl": "turtle",
        ".n3": "n3",
        ".rdf": "xml",
        ".xml": "xml",
        ".nt": "nt",
    }
    fmt = format_map.get(ext, "turtle")

    g.parse(local_path, format=fmt)
    return g


def read_file_content(url_or_path: str) -> str:
    """Read content from a URL or local path.

    Args:
        url_or_path: URL or local file path

    Returns:
        File content as string
    """
    if url_or_path.startswith("http"):
        local_path = download_file(url_or_path)
    else:
        local_path = url_or_path

    return Path(local_path).read_text(encoding="utf-8")


class W3CTestCase:
    """Represents a single W3C SPARQL evaluation test case."""

    def __init__(
        self,
        uri: URIRef,
        name: str,
        comment: Optional[str],
        query_url: str,
        data_urls: List[str],
        graph_data: List[Tuple[str, str]],  # (name, url) pairs for named graphs
        result_url: Optional[str],
        test_type: URIRef,
    ):
        self.uri = uri
        self.name = name
        self.comment = comment
        self.query_url = query_url
        self.data_urls = data_urls
        self.graph_data = graph_data
        self.result_url = result_url
        self.test_type = test_type

    def __repr__(self):
        return f"W3CTestCase({self.name!r})"


class ManifestParser:
    """Parses W3C SPARQL test manifests."""

    def __init__(self, manifest_url: str):
        self.manifest_url = manifest_url
        self.graph = Graph()

    def parse(self) -> List[W3CTestCase]:
        """Parse the manifest and return test cases."""
        # Load the main manifest
        self.graph = load_graph(self.manifest_url)

        test_cases = []

        # The manifest-evaluation.ttl includes other manifests
        # Look for mf:include to find sub-manifests
        for manifest_uri in self.graph.subjects(RDF.type, MF.Manifest):
            # Get included manifests
            includes = self.graph.value(manifest_uri, MF.include)
            if includes:
                # It's an RDF list - parse it
                for include_ref in self._parse_rdf_list(includes):
                    # Resolve the included manifest URL relative to the main manifest
                    sub_manifest_url = self._resolve_url(
                        str(include_ref), self.manifest_url
                    )
                    try:
                        sub_tests = self._parse_sub_manifest(sub_manifest_url)
                        test_cases.extend(sub_tests)
                    except Exception as e:
                        print(
                            f"Warning: Failed to parse sub-manifest {sub_manifest_url}: {e}"
                        )

            # Also get tests directly in this manifest
            entries = self.graph.value(manifest_uri, MF.entries)
            if entries:
                for test_uri in self._parse_rdf_list(entries):
                    test = self._parse_test_entry(
                        test_uri, self.graph, str(manifest_uri)
                    )
                    if test:
                        test_cases.append(test)

        return test_cases

    def _parse_sub_manifest(self, manifest_url: str) -> List[W3CTestCase]:
        """Parse a sub-manifest file.

        Args:
            manifest_url: The original HTTP URL of the manifest (used for resolving relative URLs)
        """
        # Ensure we're working with the HTTP URL for resolution purposes
        http_url = manifest_url
        if manifest_url.startswith("file://"):
            # Convert back to HTTP URL
            local_path = manifest_url.replace("file://", "")
            cache_dir_str = str(CACHE_DIR)
            if cache_dir_str in local_path:
                relative_path = local_path.replace(cache_dir_str + "/", "")
                http_url = W3C_TEST_BASE + relative_path

        g = load_graph(manifest_url)
        tests = []

        for manifest_uri in g.subjects(RDF.type, MF.Manifest):
            entries = g.value(manifest_uri, MF.entries)
            if entries:
                for test_uri in self._parse_rdf_list(entries, g):
                    # Use the HTTP URL for resolving relative URLs in test entries
                    test = self._parse_test_entry(test_uri, g, http_url)
                    if test:
                        tests.append(test)

        return tests

    def _parse_rdf_list(self, list_node, graph: Optional[Graph] = None) -> List[Any]:
        """Parse an RDF list (collection) into a Python list."""
        g = graph or self.graph
        items = []
        current = list_node

        while current and current != RDF.nil:
            first = g.value(current, RDF.first)
            if first:
                items.append(first)
            current = g.value(current, RDF.rest)

        return items

    def _parse_test_entry(
        self, test_uri: URIRef, graph: Graph, manifest_url: str
    ) -> Optional[W3CTestCase]:
        """Parse a single test entry from the manifest."""
        # Get test type
        test_type = graph.value(test_uri, RDF.type)
        if test_type != MF.QueryEvaluationTest:
            # We only care about QueryEvaluationTest for now
            return None

        # Get basic metadata
        name = str(graph.value(test_uri, MF.name) or test_uri.split("#")[-1])
        comment = graph.value(test_uri, RDFS.comment)
        if comment:
            comment = str(comment)

        # Get the action (query and data)
        action = graph.value(test_uri, MF.action)
        if not action:
            return None

        # Get query URL
        query_url = graph.value(action, QT.query)
        if not query_url:
            return None
        query_url = self._resolve_url(str(query_url), manifest_url)

        # Get default graph data
        data_urls = []
        data = graph.value(action, QT.data)
        if data:
            data_urls.append(self._resolve_url(str(data), manifest_url))

        # Get named graph data
        graph_data = []
        for gd in graph.objects(action, QT.graphData):
            if isinstance(gd, URIRef):
                graph_data.append((str(gd), self._resolve_url(str(gd), manifest_url)))
            elif isinstance(gd, BNode):
                # Named graph with explicit label
                gd_graph = graph.value(gd, QT.graph)
                gd_label = graph.value(gd, RDFS.label)
                if gd_graph:
                    name_str = str(gd_label) if gd_label else str(gd_graph)
                    graph_data.append(
                        (name_str, self._resolve_url(str(gd_graph), manifest_url))
                    )

        # Get expected result
        result = graph.value(test_uri, MF.result)
        result_url = self._resolve_url(str(result), manifest_url) if result else None

        return W3CTestCase(
            uri=test_uri,
            name=name,
            comment=comment,
            query_url=query_url,
            data_urls=data_urls,
            graph_data=graph_data,
            result_url=result_url,
            test_type=test_type,
        )

    def _resolve_url(self, url: str, base_url: str) -> str:
        """Resolve a potentially relative URL against a base URL."""
        # Handle file:// URLs from rdflib resolving against local cache
        cache_dir_str = str(CACHE_DIR)
        if url.startswith("file://") and cache_dir_str in url:
            relative_path = url.replace("file://", "").replace(cache_dir_str + "/", "")
            return W3C_TEST_BASE + relative_path

        # Absolute HTTP URLs returned as-is
        if url.startswith(("http://", "https://")):
            return url

        # Convert file:// base URLs back to HTTP
        if base_url.startswith("file://") and cache_dir_str in base_url:
            relative_base = base_url.replace("file://", "").replace(
                cache_dir_str + "/", ""
            )
            base_url = W3C_TEST_BASE + relative_base

        # Get directory of base URL and append relative URL
        if base_url.endswith((".ttl", ".rdf", ".n3")):
            base_dir = "/".join(base_url.split("/")[:-1]) + "/"
        else:
            base_dir = base_url if base_url.endswith("/") else base_url + "/"

        return base_dir + url


class SparqlResultParser:
    """Parse SPARQL query results in various formats."""

    @staticmethod
    def parse_srx(filepath: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Parse SPARQL Results XML format (.srx).

        Returns:
            Tuple of (variable_names, list_of_bindings)
        """
        import xml.etree.ElementTree as ET

        content = read_file_content(filepath)
        root = ET.fromstring(content)

        # Namespace handling
        ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}

        # Get variable names
        variables = []
        head = root.find("sparql:head", ns)
        if head is not None:
            for var in head.findall("sparql:variable", ns):
                variables.append(var.get("name"))

        # Get results
        bindings_list = []
        results = root.find("sparql:results", ns)
        if results is not None:
            for result in results.findall("sparql:result", ns):
                binding = {}
                for bind in result.findall("sparql:binding", ns):
                    var_name = bind.get("name")

                    # Check for different value types
                    uri = bind.find("sparql:uri", ns)
                    literal = bind.find("sparql:literal", ns)
                    bnode = bind.find("sparql:bnode", ns)

                    if uri is not None:
                        binding[var_name] = f"<{uri.text}>"
                    elif literal is not None:
                        # Handle datatype and language tags
                        text = literal.text or ""
                        datatype = literal.get("datatype")
                        lang = literal.get("{http://www.w3.org/XML/1998/namespace}lang")

                        if datatype:
                            binding[var_name] = f'"{text}"^^<{datatype}>'
                        elif lang:
                            binding[var_name] = f'"{text}"@{lang}'
                        else:
                            binding[var_name] = text
                    elif bnode is not None:
                        binding[var_name] = f"_:{bnode.text}"
                    else:
                        binding[var_name] = None

                bindings_list.append(binding)

        return variables, bindings_list

    @staticmethod
    def parse_ttl_results(filepath: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Parse SPARQL results stored in Turtle format.

        The W3C test suite sometimes uses the result-set RDF vocabulary.
        """
        g = load_graph(filepath)

        # Get the ResultSet
        result_set = None
        for rs in g.subjects(RDF.type, RS.ResultSet):
            result_set = rs
            break

        if not result_set:
            return [], []

        # Get variables
        variables = []
        for var in g.objects(result_set, RS.resultVariable):
            variables.append(str(var))

        # Get solutions
        bindings_list = []
        for solution in g.objects(result_set, RS.solution):
            binding = {}
            for b in g.objects(solution, RS.binding):
                var_name = str(g.value(b, RS.variable))
                value = g.value(b, RS.value)

                if isinstance(value, URIRef):
                    binding[var_name] = f"<{value}>"
                elif isinstance(value, Literal):
                    if value.datatype:
                        binding[var_name] = f'"{value}"^^<{value.datatype}>'
                    elif value.language:
                        binding[var_name] = f'"{value}"@{value.language}'
                    else:
                        binding[var_name] = str(value)
                elif isinstance(value, BNode):
                    binding[var_name] = f"_:{value}"
                else:
                    binding[var_name] = None

            bindings_list.append(binding)

        return variables, bindings_list

    @classmethod
    def parse(cls, filepath: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Parse results file, auto-detecting format."""
        ext = Path(filepath).suffix.lower()

        if ext == ".srx":
            return cls.parse_srx(filepath)
        elif ext in (".ttl", ".rdf", ".n3"):
            return cls.parse_ttl_results(filepath)
        else:
            # Try both formats
            try:
                return cls.parse_srx(filepath)
            except Exception:
                return cls.parse_ttl_results(filepath)


def normalize_value(value: Any) -> Any:
    """Normalize a value for comparison.

    Handles differences in formatting between expected and actual results.
    """
    if value is None:
        return None

    s = str(value)

    # Remove surrounding whitespace
    s = s.strip()

    # Normalize integer representations
    # e.g., "1"^^xsd:integer and "01"^^xsd:integer should match
    if "^^" in s and "integer" in s.lower():
        try:
            # Extract the value part
            val_part = s.split("^^")[0].strip('"')
            return int(val_part)
        except ValueError:
            pass

    return s


def results_equivalent(
    actual_vars: List[str],
    actual_rows: List[Dict[str, Any]],
    expected_vars: List[str],
    expected_rows: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Check if two result sets are equivalent.

    SPARQL results are sets (unordered) of solutions.
    Variables must match, and the multiset of bindings must match.

    Returns:
        Tuple of (is_equivalent, explanation_if_not)
    """
    # Check variables match (order doesn't matter)
    actual_var_set = set(actual_vars)
    expected_var_set = set(expected_vars)

    if actual_var_set != expected_var_set:
        return (
            False,
            f"Variable mismatch: actual={actual_var_set}, expected={expected_var_set}",
        )

    # Check row counts
    if len(actual_rows) != len(expected_rows):
        return (
            False,
            f"Row count mismatch: actual={len(actual_rows)}, expected={len(expected_rows)}",
        )

    # Normalize and compare as multisets
    def normalize_row(row: Dict[str, Any], vars: List[str]) -> tuple:
        """Convert a row to a normalized tuple for comparison."""
        return tuple(normalize_value(row.get(v)) for v in sorted(vars))

    actual_multiset = sorted([normalize_row(r, actual_vars) for r in actual_rows])
    expected_multiset = sorted([normalize_row(r, expected_vars) for r in expected_rows])

    if actual_multiset != expected_multiset:
        return (
            False,
            f"Row content mismatch:\nActual: {actual_multiset}\nExpected: {expected_multiset}",
        )

    return True, ""


def create_test_method(test_case: W3CTestCase):
    """Create a test method for a W3C test case.

    Tests are skipped automatically when:
    - Test data cannot be loaded
    - Query cannot be read
    - The translator raises NotImplementedError (unsupported feature)
    - Expected results cannot be parsed
    - No result URL is provided
    """

    def test_method(self):
        # Skip tests without result files
        if not test_case.result_url:
            self.skipTest("No result URL provided")

        # Load test data into the database
        try:
            self._load_test_data(test_case)
        except Exception as e:
            self.skipTest(f"Failed to load test data: {e}")

        # Read the query
        try:
            query = read_file_content(test_case.query_url)
        except Exception as e:
            self.skipTest(f"Failed to read query: {e}")

        # Execute the query through our translator
        # NotImplementedError means the feature isn't supported yet - skip the test
        try:
            result = self.translator.execute(query)
            actual_rows = result.fetchall()
        except NotImplementedError as e:
            self.skipTest(f"Feature not implemented: {e}")
        except Exception as e:
            self.fail(f"Query execution failed: {e}")

        # Parse expected results
        try:
            expected_vars, expected_bindings = SparqlResultParser.parse(
                test_case.result_url
            )
        except Exception as e:
            self.skipTest(f"Failed to parse expected results: {e}")

        # Convert actual results to comparable format
        actual_vars = list(actual_rows[0]._fields) if actual_rows else []
        actual_bindings = [dict(zip(actual_vars, row)) for row in actual_rows]

        # Compare results
        is_eq, reason = results_equivalent(
            actual_vars, actual_bindings, expected_vars, expected_bindings
        )

        if not is_eq:
            self.fail(f"Results mismatch for {test_case.name}: {reason}")

    # Set test method name and docstring
    test_method.__name__ = f"test_{_sanitize_name(test_case.name)}"
    test_method.__doc__ = test_case.comment or f"W3C test: {test_case.name}"

    return test_method


def _sanitize_name(name: str) -> str:
    """Sanitize a test name to be a valid Python identifier."""
    import re

    # Replace non-alphanumeric chars with underscore
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    # Remove leading digits
    name = re.sub(r"^[0-9]+", "", name)
    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)
    # Remove trailing underscores
    name = name.strip("_")
    return name or "unnamed"


class W3CSparqlTestBase(unittest.TestCase):
    """Base class for W3C SPARQL tests."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        # Create in-memory SQLite database
        cls.engine = create_engine("sqlite:///:memory:")
        cls.translator = AlgebraTranslator(
            cls.engine, table_name="triples", create_table=True
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level resources."""
        if hasattr(cls, "engine"):
            cls.engine.dispose()

    def setUp(self):
        """Clear the database before each test."""
        with self.engine.connect() as conn:
            conn.execute(self.translator.table.delete())
            conn.commit()

    def _load_test_data(self, test_case: W3CTestCase):
        """Load test data into the database."""
        # Load default graph data
        for data_url in test_case.data_urls:
            g = load_graph(data_url)
            self._load_graph_to_db(g)

        # Note: Named graphs are not fully supported in our simple triple store model
        # For now, we merge named graph data into the default graph
        for name, url in test_case.graph_data:
            g = load_graph(url)
            self._load_graph_to_db(g)

    def _load_graph_to_db(self, g: Graph):
        """Load an rdflib Graph into the database."""
        triples = []
        for s, p, o in g:
            s_str = term_to_string(s) or str(s)
            p_str = term_to_string(p) or str(p)

            if isinstance(o, URIRef):
                o_str = f"<{o}>"
            elif isinstance(o, Literal):
                # Format literals according to our storage format
                if o.datatype:
                    o_str = f'"{o}"^^<{o.datatype}>'
                elif o.language:
                    o_str = f'"{o}"@{o.language}'
                else:
                    o_str = str(o)
            elif isinstance(o, BNode):
                o_str = f"_:{o}"
            else:
                o_str = str(o)

            triples.append({"s": s_str, "p": p_str, "o": o_str})

        if triples:
            with self.engine.connect() as conn:
                conn.execute(self.translator.table.insert(), triples)
                conn.commit()


class W3CSparqlTests(W3CSparqlTestBase):
    """W3C SPARQL evaluation tests loaded dynamically from the manifest."""

    pass


def load_tests(loader, tests, pattern):
    """Load W3C SPARQL evaluation tests dynamically."""
    suite = unittest.TestSuite()

    try:
        parser = ManifestParser(MANIFEST_URL)
        test_cases = parser.parse()

        for tc in test_cases:
            method = create_test_method(tc)
            setattr(W3CSparqlTests, method.__name__, method)

        suite.addTests(loader.loadTestsFromTestCase(W3CSparqlTests))

    except Exception as e:

        class W3CTestLoadError(unittest.TestCase):
            def test_load_manifest(self):
                self.skipTest(f"Failed to load W3C test manifest: {e}")

        suite.addTests(loader.loadTestsFromTestCase(W3CTestLoadError))

    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
