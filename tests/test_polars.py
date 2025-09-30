import unittest
import polars as pl
import rdfdf as pr

from unittest.mock import patch
from polars.testing import assert_frame_equal

import rdfdf._polars as rdf
import rdfdf.pattern as pat
from rdfdf.pattern import V

class TestPolars(unittest.TestCase):

    def setUp(self):
        self.triples = pl.DataFrame(
            {
                "s": ["s1", "s2", "s3"],
                "p": ["p1", "p2", "p3"],
                "o": ["o1", "o2", "s1"],
            }
        )

    def test_vars(self):
        v1, v2 = pat.vars("v1", "v2")
        self.assertEqual(v1.name(), "v1")
        self.assertEqual(v2.name(), "v2")
        self.assertEqual(pat.V.a, pat.V.a)
        self.assertNotEqual(v1._unique_name(), v2._unique_name())
        self.assertNotEqual(pat.V.a, pat.V.b)

    def test_triple_pattern_to_df(self):
        r = rdf.triple_pattern_to_df((pat.IRI("s1"), V.a, V.b), self.triples)
        expected = pl.DataFrame([["p1", "o1"]], schema=list("ab"), orient="row")
        assert_frame_equal(r.collect(), expected)

    def test_graph_pattern_to_df(self):
        r = rdf.graph_pattern_to_df([(V.a, V.b, V.c),
                                     (V.d, V.e, V.a)], self.triples)
        expected = pl.DataFrame([["s1", "p1", "o1", "s3", "p3"]], schema=list("abcde"), orient="row")
        assert_frame_equal(r.collect(), expected,)

    def test_select_mgq(self):
        s, p, o = pr.vars(*"spo")
        res = pr.select(self.triples, [s, p, o], [[s, p, o]])
        assert_frame_equal(self.triples, res.collect())

    def test_select_with_expr(self):
        p, o = pr.vars(*"po")
        res = pr.select(self.triples, [pl.lit("22").alias("L"), p, o], [["s1", p, o]])
        expect = pl.DataFrame({"L": "22", "p": "p1", "o": "o1"})
        assert_frame_equal(expect, res.collect())

    def test_select_s(self):
        p, o = pr.vars(*"po")
        res = pr.select(self.triples, [p, o], [["s1", p, o]])
        expect = pl.DataFrame({"p": "p1", "o": "o1"})
        assert_frame_equal(expect, res.collect())

    def test_multiple_clauses(self):
        s, p, o = pr.vars(*"spo")
        x, y, z = pr.vars(*"xyz")
        res = pr.select(self.triples, [s, p, o, y, z], [(s, p, o), (o, y, z)])
        expect = pl.DataFrame({"s": "s3", "p": "p3", "o": "s1", "y": "p1", "z": "o1"})
        assert_frame_equal(expect, res.collect())

    def test_select_optional(self):
        unittest.skip("TODO")

    def test_select_filter(self):
        unittest.skip("TODO")

    def test_construct(self):
        unittest.skip("TODO")

    def test_from_df(self):
        unittest.skip("TODO")

    def test_is_variable(self):
        (v1,) = pr.vars("v1")
        self.assertTrue(pr.is_variable(v1))
        self.assertFalse(pr.is_variable("literal"))

    def test_is_literal(self):
        self.assertTrue(pr.is_literal("literal"))
        self.assertFalse(pr.is_literal(pr.vars(1)))

    def test_multiway_natural_join(self):
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pl.DataFrame({"a": [1, 2, 3], "c": [5, 6, 7]})
        df = pr._multiway_natural_join(df1, df2)
        self.assertEqual(df.shape, (2, 3))
        assert_frame_equal(df, pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}))

    def test_expand_pattern_po(self):
        unittest.skip("TODO")

    def test_expand_pattern_s(self):
        unittest.skip("TODO")

    def test_expand_pattern_p(self):
        with patch("rdfdf.vars", side_effect=lambda n: list(range(n))) as mock:
            pattern = ("s", ["p1", "p2", "p3"], "o")
            expanded = list(pr._expand_pattern(pattern))
        self.assertEqual(expanded, [('s', 'p1', 0), (0, 'p2', 1), (1, 'p3', 'o')])

    def test_expand_pattern_o(self):
        unittest.skip("TODO")


if __name__ == "__main__":
    unittest.main()
