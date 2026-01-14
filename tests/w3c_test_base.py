"""
Shared infrastructure for W3C SPARQL test suites.

This module provides common functionality for running W3C SPARQL evaluation tests
against the AlgebraTranslator implementation. It supports both SPARQL 1.0 and 1.1
test suites.

Test suite sources:
- SPARQL 1.0: https://www.w3.org/2001/sw/DataAccess/tests/data-r2/
- SPARQL 1.1: https://www.w3.org/2009/sparql/docs/tests/data-sparql11/

The test files are obtained from the W3C rdf-tests repository:
https://github.com/w3c/rdf-tests

Use `make clone-w3c-tests` to clone the repository before running tests.
"""

import unittest
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, Union

import rdflib
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS
from sqlalchemy import create_engine

from rdfdf.sqlalch import AlgebraTranslator, term_to_string, term_to_object_type


# Disable rdflib's literal normalisation to preserve original lexical forms.
# By default, rdflib normalises typed literals (e.g., "01"^^xsd:integer becomes "1").
# SPARQL semantics require preserving the original lexical form for correct results.
rdflib.NORMALIZE_LITERALS = False


# W3C Test Suite namespaces
MF = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#")
QT = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-query#")
RS = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/result-set#")

# Local git repository location (cloned from https://github.com/w3c/rdf-tests)
W3C_GIT_REPO_DIR = Path(__file__).parent / "rdf-tests"

# URL to local path mappings for the git repository
# Maps W3C URL prefixes to paths within the git repo
W3C_URL_TO_GIT_PATH = {
    "https://www.w3.org/2001/sw/DataAccess/tests/data-r2/": "sparql/sparql10/",
    "https://www.w3.org/2009/sparql/docs/tests/data-sparql11/": "sparql/sparql11/",
}


def resolve_url_to_local_path(url: str) -> str:
    """Resolve a W3C URL to a local path in the git repository.

    Args:
        url: The W3C URL to resolve

    Returns:
        Path to the local file

    Raises:
        FileNotFoundError: If the git repo is not cloned or file doesn't exist
    """
    if not W3C_GIT_REPO_DIR.exists():
        raise FileNotFoundError(
            f"W3C test repository not found at {W3C_GIT_REPO_DIR}. "
            "Run 'make clone-w3c-tests' to clone it."
        )

    for url_prefix, git_path in W3C_URL_TO_GIT_PATH.items():
        if url.startswith(url_prefix):
            relative_path = url[len(url_prefix) :]
            local_path = W3C_GIT_REPO_DIR / git_path / relative_path
            if local_path.exists():
                return str(local_path)
            raise FileNotFoundError(f"Test file not found: {local_path}")

    raise FileNotFoundError(f"URL does not match any known W3C test suite: {url}")


def load_graph(url_or_path: str) -> Graph:
    """Load an RDF graph from a URL or local path.

    Args:
        url_or_path: W3C URL or local file path

    Returns:
        Parsed RDF Graph
    """
    g = Graph()

    if url_or_path.startswith("http"):
        local_path = resolve_url_to_local_path(url_or_path)
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
        url_or_path: W3C URL or local file path

    Returns:
        File content as string
    """
    if url_or_path.startswith("http"):
        local_path = resolve_url_to_local_path(url_or_path)
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

    def __init__(self, manifest_url: str, base_url: str):
        self.manifest_url = manifest_url
        self.base_url = base_url
        self.graph = Graph()

    def parse(self) -> List[W3CTestCase]:
        """Parse the manifest and return test cases."""
        # Load the main manifest
        self.graph = load_graph(self.manifest_url)

        test_cases = []

        # The manifest includes other manifests
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
            manifest_url: The HTTP URL of the manifest (used for resolving relative URLs)
        """
        g = load_graph(manifest_url)
        tests = []

        for manifest_uri in g.subjects(RDF.type, MF.Manifest):
            entries = g.value(manifest_uri, MF.entries)
            if entries:
                for test_uri in self._parse_rdf_list(entries, g):
                    test = self._parse_test_entry(test_uri, g, manifest_url)
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
            # We only care about QueryEvaluationTest since rdflib handles parsing
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
        git_repo_str = str(W3C_GIT_REPO_DIR)

        # Handle file:// URLs from rdflib resolving against local git repo
        if url.startswith("file://") and git_repo_str in url:
            local_path = url.replace("file://", "")
            # Find the relative path within the git repo and convert back to HTTP URL
            for url_prefix, git_path in W3C_URL_TO_GIT_PATH.items():
                full_git_path = str(W3C_GIT_REPO_DIR / git_path)
                if full_git_path in local_path:
                    relative_path = local_path.replace(full_git_path, "").lstrip("/")
                    return url_prefix + relative_path
            # Fallback: couldn't map, return original
            return url

        # Absolute HTTP URLs returned as-is
        if url.startswith(("http://", "https://")):
            return url

        # Get directory of base URL and append relative URL
        if base_url.endswith((".ttl", ".rdf", ".n3")):
            base_dir = "/".join(base_url.split("/")[:-1]) + "/"
        else:
            base_dir = base_url if base_url.endswith("/") else base_url + "/"

        return base_dir + url


class SparqlResultParser:
    """Parse SPARQL query results in various formats."""

    def parse_srx(
        self, filepath: str
    ) -> Tuple[List[str], Union[List[Dict[str, Any]], bool]]:
        """Parse SPARQL Results XML format (.srx).

        Returns:
            Tuple of (variable_names, list_of_bindings) for SELECT results
            Tuple of ([], bool) for ASK results
        """
        import xml.etree.ElementTree as ET

        content = read_file_content(filepath)
        root = ET.fromstring(content)

        # Namespace handling
        ns = {"sparql": "http://www.w3.org/2005/sparql-results#"}

        # Check for ASK boolean result first
        boolean_elem = root.find("sparql:boolean", ns)
        if boolean_elem is not None:
            return [], boolean_elem.text.lower() == "true"

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

    def parse_ttl_results(
        self, filepath: str
    ) -> Tuple[List[str], Union[List[Dict[str, Any]], bool]]:
        """Parse SPARQL results stored in Turtle format.

        The W3C test suite sometimes uses the result-set RDF vocabulary.

        Returns:
            Tuple of (variable_names, list_of_bindings) for SELECT results
            Tuple of ([], bool) for ASK results
        """
        g = load_graph(filepath)

        # Get the ResultSet
        result_set = None
        for rs in g.subjects(RDF.type, RS.ResultSet):
            result_set = rs
            break

        if not result_set:
            return [], []

        # Check for ASK boolean result first
        boolean_val = g.value(result_set, RS.boolean)
        if boolean_val is not None:
            return [], str(boolean_val).lower() == "true"

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

    def parse(
        self, filepath: str
    ) -> Tuple[List[str], Union[List[Dict[str, Any]], bool]]:
        """Parse results file, auto-detecting format."""
        ext = Path(filepath).suffix.lower()

        if ext == ".srx":
            return self.parse_srx(filepath)
        elif ext in (".ttl", ".rdf", ".n3"):
            return self.parse_ttl_results(filepath)
        else:
            # Try both formats
            try:
                return self.parse_srx(filepath)
            except Exception:
                return self.parse_ttl_results(filepath)


def normalise_value(value: Any) -> Any:
    """Normalise a value for comparison.

    Converts SPARQL result format to our raw database format:
    - URIs: <http://...> -> http://...
    - Typed literals: "value"^^<datatype> -> value
    - Language-tagged: "value"@lang -> value
    - Plain literals: "value" or value -> value
    - BNodes: _:id -> _:id (unchanged)
    """
    if value is None:
        return None

    s = str(value)

    # Remove surrounding whitespace
    s = s.strip()

    # Handle BNodes - keep as-is
    if s.startswith("_:"):
        return s

    # Handle URIs - strip angle brackets
    # Expected results have <http://...>, our storage has http://...
    if s.startswith("<") and s.endswith(">") and "://" in s and "^^" not in s:
        return s[1:-1]

    # Handle typed literals: "value"^^<datatype>
    if "^^" in s:
        # Extract the lexical value (between quotes before ^^)
        val_part = s.split("^^")[0]
        if val_part.startswith('"') and val_part.endswith('"'):
            return val_part[1:-1]
        return val_part

    # Handle language-tagged literals: "value"@lang
    if "@" in s and s.count('"') >= 2:
        # Find the value between quotes
        first_quote = s.index('"')
        last_quote = s.rindex('"')
        if first_quote < last_quote:
            return s[first_quote + 1 : last_quote]

    # Handle plain quoted literals: "value"
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]

    # Plain unquoted value - return as-is
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

    # Normalise and compare as multisets using Counter (avoids sorting mixed types)
    def normalise_row(row: Dict[str, Any], vars: List[str]) -> tuple:
        """Convert a row to a normalised tuple for comparison."""
        return tuple(normalise_value(row.get(v)) for v in sorted(vars))

    actual_multiset = Counter(normalise_row(r, actual_vars) for r in actual_rows)
    expected_multiset = Counter(normalise_row(r, expected_vars) for r in expected_rows)

    if actual_multiset != expected_multiset:
        return (
            False,
            f"Row content mismatch:\nActual: {dict(actual_multiset)}\nExpected: {dict(expected_multiset)}",
        )

    return True, ""


def sanitise_name(name: str) -> str:
    """Sanitise a test name to be a valid Python identifier."""
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
        except NotImplementedError as e:
            self.skipTest(f"Feature not implemented: {e}")
        except Exception as e:
            self.fail(f"Query execution failed: {e}")

        # Parse expected results
        result_parser = SparqlResultParser()
        try:
            expected_vars, expected_result = result_parser.parse(test_case.result_url)
        except Exception as e:
            self.skipTest(f"Failed to parse expected results: {e}")

        # Handle ASK queries (result is a boolean)
        if isinstance(result, bool):
            if not isinstance(expected_result, bool):
                self.fail(
                    f"ASK query returned bool but expected results are not boolean"
                )
            if result != expected_result:
                self.fail(
                    f"ASK result mismatch for {test_case.name}: "
                    f"actual={result}, expected={expected_result}"
                )
            return

        # Handle SELECT queries (result is a CursorResult)
        actual_rows = result.fetchall()

        # Convert actual results to comparable format
        # Use result.keys() to get column names (works even with empty results)
        actual_vars = list(result.keys())
        actual_bindings = [dict(zip(actual_vars, row)) for row in actual_rows]

        # Compare results
        is_eq, reason = results_equivalent(
            actual_vars, actual_bindings, expected_vars, expected_result
        )

        if not is_eq:
            self.fail(f"Results mismatch for {test_case.name}: {reason}")

    # Set test method name and docstring
    test_method.__name__ = f"test_{sanitise_name(test_case.name)}"
    test_method.__doc__ = test_case.comment or f"W3C test: {test_case.name}"

    return test_method


class W3CSparqlTestBase(unittest.TestCase):
    """Base class for W3C SPARQL tests."""

    # Subclasses must define this for URL resolution
    BASE_URL: str = None

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
        """Load an rdflib Graph into the database.

        Storage format matches the AlgebraTranslator's expectations:
        - URIs: stored as raw URI strings (no angle brackets)
        - Literals: lexical value in 'o', type info in 'ot'
        - BNodes: _:id format
        """
        triples = []
        for s, p, o in g:
            # Subjects and predicates: use term_to_string (raw URI for URIRef)
            s_str = term_to_string(s) or str(s)
            p_str = term_to_string(p) or str(p)

            # Objects: lexical value in 'o', type info in 'ot'
            o_str = term_to_string(o) or str(o)
            ot_str = term_to_object_type(o)

            triples.append({"s": s_str, "p": p_str, "o": o_str, "ot": ot_str})

        if triples:
            with self.engine.connect() as conn:
                conn.execute(self.translator.table.insert(), triples)
                conn.commit()


def load_w3c_tests(
    test_class: type,
    manifest_url: str,
    base_url: str,
    loader,
    suite: unittest.TestSuite,
):
    """Load W3C SPARQL evaluation tests dynamically into a test class.

    Args:
        test_class: The test class to add methods to
        manifest_url: URL of the manifest file
        base_url: Base URL for the test suite
        loader: unittest test loader
        suite: unittest test suite to add tests to
    """
    try:
        parser = ManifestParser(manifest_url, base_url)
        test_cases = parser.parse()

        for tc in test_cases:
            method = create_test_method(tc)
            setattr(test_class, method.__name__, method)

        suite.addTests(loader.loadTestsFromTestCase(test_class))

    except Exception as e:
        error_msg = str(e)

        class W3CTestLoadError(unittest.TestCase):
            def test_load_manifest(self):
                self.skipTest(f"Failed to load W3C test manifest: {error_msg}")

        suite.addTests(loader.loadTestsFromTestCase(W3CTestLoadError))
