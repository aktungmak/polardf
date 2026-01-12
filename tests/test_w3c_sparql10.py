"""
W3C SPARQL 1.0 Evaluation Test Suite for sqlalch.py

This module runs the W3C SPARQL 1.0 evaluation test cases against the
AlgebraTranslator implementation to verify conformance with the SPARQL spec.

Test suite source: https://www.w3.org/2001/sw/DataAccess/tests/data-r2/
Manifest: https://www.w3.org/2001/sw/DataAccess/tests/data-r2/manifest-evaluation.ttl

Test files are from the W3C rdf-tests repository:
https://github.com/w3c/rdf-tests

Run `make clone-w3c-tests` to clone the repository before running tests.

We only run evaluation tests (not syntax tests since rdflib handles parsing)
and filter for SELECT queries since that's what's currently implemented.
"""

import unittest

from tests.w3c_test_base import (
    W3CSparqlTestBase,
    load_w3c_tests,
)


# SPARQL 1.0 test suite configuration
W3C_TEST_BASE = "https://www.w3.org/2001/sw/DataAccess/tests/data-r2/"
MANIFEST_URL = W3C_TEST_BASE + "manifest-evaluation.ttl"


class W3CSparqlTests(W3CSparqlTestBase):
    """W3C SPARQL 1.0 evaluation tests loaded dynamically from the manifest."""

    BASE_URL = W3C_TEST_BASE


def load_tests(loader, tests, pattern):
    """Load W3C SPARQL 1.0 evaluation tests dynamically."""
    suite = unittest.TestSuite()
    load_w3c_tests(W3CSparqlTests, MANIFEST_URL, W3C_TEST_BASE, loader, suite)
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
