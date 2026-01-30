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

_Last updated: 2026-01-30 20:26:11_

| Suite | Passed | Failed | Errors | Skipped | Total |
|-------|--------|--------|--------|---------|-------|
| SPARQL 1.0 | 236 | 26 | 0 | 13 | 275 |
| SPARQL 1.1 | 103 | 83 | 0 | 116 | 302 |
| **Total** | **339** | **109** | **0** | **129** | **577** |

**Pass rate: 58.8%** (339/577 tests)
