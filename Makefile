.PHONY: test test-sqlalch test-w3c test-w3c-10 test-w3c-11 clone-w3c-tests

# W3C RDF tests repository location
W3C_TESTS_DIR = tests/rdf-tests
W3C_TESTS_REPO = https://github.com/w3c/rdf-tests.git

test: test-sqlalch test-w3c

test-sqlalch:
	./venv/bin/python -m unittest tests.test_sqlalch -v

# Run all W3C tests (both SPARQL 1.0 and 1.1)
test-w3c: test-w3c-10 test-w3c-11

# Clone or update W3C rdf-tests repository (much faster than HTTP downloads)
clone-w3c-tests:
	@if [ -d "$(W3C_TESTS_DIR)" ]; then \
		echo "W3C tests repo already exists, pulling latest..."; \
		cd $(W3C_TESTS_DIR) && git pull --quiet; \
	else \
		echo "Cloning W3C rdf-tests repository..."; \
		git clone --depth 1 $(W3C_TESTS_REPO) $(W3C_TESTS_DIR); \
	fi

# SPARQL 1.0 tests (data-r2)
test-w3c-10: clone-w3c-tests
	./venv/bin/python -m unittest tests.test_w3c_sparql10 -v

# SPARQL 1.1 tests (data-sparql11) - includes property paths, aggregates, etc.
test-w3c-11: clone-w3c-tests
	./venv/bin/python -m unittest tests.test_w3c_sparql11 -v
