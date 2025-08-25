import unittest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
import sys
import os

import rdfdf.pattern

# Add the parent directory to the path to import rdfdf modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rdfdf import spark as pr_spark
from rdfdf.pattern import vars
from unittest.mock import patch


def assert_dataframe_equal(df1, df2):
    """Helper function to compare PySpark DataFrames"""
    rows1 = df1.collect()
    rows2 = df2.collect()
    
    # Check if they have the same number of rows
    if len(rows1) != len(rows2):
        raise AssertionError(f"DataFrames have different number of rows: {len(rows1)} vs {len(rows2)}")
    
    # Check if they have the same columns
    if set(df1.columns) != set(df2.columns):
        raise AssertionError(f"DataFrames have different columns: {df1.columns} vs {df2.columns}")
    
    # Sort both DataFrames by all columns for consistent comparison
    df1_sorted = df1.orderBy(*df1.columns).collect()
    df2_sorted = df2.orderBy(*df2.columns).collect()
    
    # Compare row by row
    for i, (row1, row2) in enumerate(zip(df1_sorted, df2_sorted)):
        if row1.asDict() != row2.asDict():
            raise AssertionError(f"DataFrames differ at row {i}: {row1.asDict()} vs {row2.asDict()}")


class TestSpark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up Spark session for all tests"""
        cls.spark = SparkSession.builder \
            .appName("TestSparkRDF") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", "1") \
            .getOrCreate()
        cls.spark.sparkContext.setLogLevel("WARN")

    @classmethod
    def tearDownClass(cls):
        """Clean up Spark session"""
        cls.spark.stop()

    def setUp(self):
        self.triples = self.spark.createDataFrame([
            ("s1", "p1", "o1"),
            ("s2", "p2", "o2"),
            ("s3", "p3", "s1")
        ], ["s", "p", "o"])

    def test_select_mgq(self):
        s, p, o = vars(*"spo")
        res = pr_spark.select(self.triples, [s, p, o], [[s, p, o]])
        assert_dataframe_equal(self.triples, res)

    def test_select_with_expr(self):
        p, o = vars(*"po")
        res = pr_spark.select(self.triples, [F.lit("22").alias("L"), p, o], [["s1", p, o]])
        expect = self.spark.createDataFrame([("22", "p1", "o1")], ["L", "p", "o"])
        assert_dataframe_equal(expect, res)

    def test_select_s(self):
        p, o = vars(*"po")
        res = pr_spark.select(self.triples, [p, o], [["s1", p, o]])
        expect = self.spark.createDataFrame([("p1", "o1")], ["p", "o"])
        assert_dataframe_equal(expect, res)

    def test_multiple_clauses(self):
        s, p, o = vars(*"spo")
        x, y, z = vars(*"xyz")
        res = pr_spark.select(self.triples, [s, p, o, y, z], [(s, p, o), (o, y, z)])
        expect = self.spark.createDataFrame([("s3", "p3", "s1", "p1", "o1")], ["s", "p", "o", "y", "z"])
        assert_dataframe_equal(expect, res)

    def test_select_optional(self):
        self.skipTest("TODO")

    def test_select_filter(self):
        self.skipTest("TODO")

    def test_construct(self):
        self.skipTest("TODO")

    def test_from_df(self):
        self.skipTest("TODO")

    def test_is_variable(self):
        (v1,) = vars("v1")
        self.assertTrue(rdfdf.pattern.is_variable(v1))
        self.assertFalse(rdfdf.pattern.is_variable("literal"))

    def test_is_literal(self):
        self.assertTrue(rdfdf.pattern.is_literal("literal"))
        self.assertFalse(rdfdf.pattern.is_literal(vars(1)))

    def test_multiway_natural_join(self):
        df1 = self.spark.createDataFrame([(1, 3), (2, 4)], ["a", "b"])
        df2 = self.spark.createDataFrame([(1, 5), (2, 6), (3, 7)], ["a", "c"])
        df = pr_spark._multiway_natural_join(df1, df2)
        self.assertEqual(df.count(), 2)
        expect = self.spark.createDataFrame([(1, 3, 5), (2, 4, 6)], ["a", "b", "c"])
        assert_dataframe_equal(df, expect)

    # TODO move to a separate testing module
    def test_expand_pattern_p(self):
        with patch("rdfdf.pattern.vars", side_effect=lambda n: list(range(n))) as mock:
            pattern = ("s", ["p1", "p2", "p3"], "o")
            expanded = list(rdfdf.pattern._expand_pattern(pattern))
        self.assertEqual(expanded, [('s', 'p1', 0), (0, 'p2', 1), (1, 'p3', 'o')])



if __name__ == "__main__":
    unittest.main() 