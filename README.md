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

_Last updated: 2026-01-27 17:50:07_

| Suite | Passed | Failed | Errors | Skipped | Total |
|-------|--------|--------|--------|---------|-------|
| SPARQL 1.0 | 219 | 25 | 0 | 20 | 264 |
| SPARQL 1.1 | 85 | 87 | 0 | 126 | 298 |
| **Total** | **304** | **112** | **0** | **146** | **562** |

**Pass rate: 73.1%** (304/416 executed tests)
