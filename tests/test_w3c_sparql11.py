"""
W3C SPARQL 1.1 Evaluation Test Suite for sqlalch.py

This module runs the W3C SPARQL 1.1 evaluation test cases against the
AlgebraTranslator implementation to verify conformance with the SPARQL 1.1 spec.

Test suite source: https://www.w3.org/2009/sparql/docs/tests/data-sparql11/
Manifest: https://www.w3.org/2009/sparql/docs/tests/data-sparql11/manifest-all.ttl

Test files are from the W3C rdf-tests repository:
https://github.com/w3c/rdf-tests

Run `make clone-w3c-tests` to clone the repository before running tests.

We only run evaluation tests (not syntax tests since rdflib handles parsing).
"""

import unittest

from tests.w3c_test_base import (
    W3CSparqlTestBase,
    load_w3c_tests,
)


# SPARQL 1.1 test suite configuration
W3C_TEST_BASE_11 = "https://www.w3.org/2009/sparql/docs/tests/data-sparql11/"
MANIFEST_URL_11 = W3C_TEST_BASE_11 + "manifest-all.ttl"


class W3CSparql11Tests(W3CSparqlTestBase):
    """W3C SPARQL 1.1 evaluation tests loaded dynamically from the manifest."""

    BASE_URL = W3C_TEST_BASE_11


def load_tests(loader, tests, pattern):
    """Load W3C SPARQL 1.1 evaluation tests dynamically."""
    suite = unittest.TestSuite()
    load_w3c_tests(W3CSparql11Tests, MANIFEST_URL_11, W3C_TEST_BASE_11, loader, suite)
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
