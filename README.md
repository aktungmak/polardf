## polardf

This library implements a dataframe version of the SPARQL query language.
It enables querying of Polars dataframes containing RDF triples using python syntax
and integrates with the rest of the Polars ecosystem by returning DataFrames.

### Features

- SELECT and CONSTRUCT queries over dataframes containing triples
- Graph patterns are specified as python datastructures, enabling
manipulation and reuse using python
- Good performance thanks to Polars' fast joins