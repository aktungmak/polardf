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

The library is tested against the official W3C test suites to validate
feature coverage and correctness, see the latest coverage below.

### Alternatives
[rdflib-sqlalchemy](https://github.com/RDFLib/rdflib-sqlalchemy)
is a great alternative however it stores triples in a highly
normalised format which makes it hard for other tools to prepare
data for querying.

### W3C Test Suite Results

_Last updated: 2026-01-29 14:42:29_

| Suite | Passed | Failed | Errors | Skipped | Total |
|-------|--------|--------|--------|---------|-------|
| SPARQL 1.0 | 231 | 26 | 0 | 18 | 275 |
| SPARQL 1.1 | 88 | 87 | 0 | 127 | 302 |
| **Total** | **319** | **113** | **0** | **145** | **577** |

**Pass rate: 55.3%** (319/577 tests)
