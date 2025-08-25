## rdfdf

This library implements a dataframe-based query language over RDF data that
provides many of the same facilities as SPARQL.
It enables querying of Spark or Polars dataframes containing RDF triples using
python syntax and integrates with the rest of the Spark/Polars ecosystem by
returning DataFrames.

### Features

- SELECT and CONSTRUCT queries over dataframes containing triples
- Graph patterns are specified as python datastructures, enabling
manipulation and reuse using python
- Good performance thanks to Polars' fast joins
- Able to handle large numbers of triples via Spark