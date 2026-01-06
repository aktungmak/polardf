.PHONY: test test-sqlalch test-w3c

test: test-sqlalch test-w3c

test-sqlalch:
	./venv/bin/python -m unittest tests.test_sqlalch -v

test-w3c:
	./venv/bin/python -m unittest tests.test_w3c_sparql -v
