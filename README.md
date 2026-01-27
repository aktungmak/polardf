## sparql2sql

This library translates SPARQL queries into SQL statements over
a simple triple table structure. This makes it simple for other
tools to populate the data since it does not require normalised
tables. As an example, [spark-r2r](github.com/aktungmak/spark-r2r)
shows how to convert tabluar data into triple format on Spark.

SQLAlchemy is used to generate the SQL code so many backends are
possible, however the following are actively tested:

- SQLite
- Postgres
- Databricks

The library is tested against the official W3C test suites and
currently has a coverage of around 60%.

### Alternatives
[rdflib-sqlalchemy](https://github.com/RDFLib/rdflib-sqlalchemy)
is a great alternative however it stores triples in a highly
normalised format which makes it hard for other tools to prepare
data for querying.

### W3C Test Suite Results

_Last updated: 2026-01-27 18:30:42_

| Suite | Passed | Failed | Errors | Skipped | Total |
|-------|--------|--------|--------|---------|-------|
| SPARQL 1.0 | 229 | 26 | 0 | 20 | 275 |
| SPARQL 1.1 | 88 | 87 | 0 | 127 | 302 |
| **Total** | **317** | **113** | **0** | **147** | **577** |

**Pass rate: 54.9%** (317/577 tests)
