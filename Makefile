.PHONY: test

test:
	./venv/bin/python -m unittest discover -s tests -v
