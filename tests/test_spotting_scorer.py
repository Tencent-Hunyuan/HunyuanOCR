"""Regression tests for the ``format3`` spotting parser.

Run from the repository root::

    python -m unittest tests/test_spotting_scorer.py -v

``reward.task_scorers.__init__`` eagerly imports every scorer, which drags in
the *parsing* scorer's heavy dependencies (pandas / bs4 /
table_recognition_metric). These tests only exercise the spotting scorer, so
the two modules it needs are loaded straight from their files; the only
third-party requirement left is ``Levenshtein``.
"""

import importlib.util
import os
import sys
import types
import unittest

_TASK_SCORERS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train_verl",
    "reward",
    "task_scorers",
)


def _load_spotting():
    pkg = types.ModuleType("_hyocr_ts")
    pkg.__path__ = [_TASK_SCORERS]
    sys.modules["_hyocr_ts"] = pkg
    for name in ("text_metrics", "spotting"):
        spec = importlib.util.spec_from_file_location(
            "_hyocr_ts." + name, os.path.join(_TASK_SCORERS, name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["_hyocr_ts." + name] = module
        spec.loader.exec_module(module)
    return sys.modules["_hyocr_ts.spotting"]


spotting = _load_spotting()


class TestParseFormat3(unittest.TestCase):
    def test_text_without_parenthesis_is_unchanged(self):
        self.assertEqual(
            spotting._parse_format3("Hello(1,2),(3,4)World(5,6),(7,8)"),
            [("Hello", 1, 2, 3, 4), ("World", 5, 6, 7, 8)],
        )

    def test_text_containing_parenthesis_is_kept(self):
        # "ATLANTA (AP)" used to make the whole item fail to match, silently
        # dropping both the text and its box from prediction and reference.
        self.assertEqual(
            spotting._parse_format3(
                "ATLANTA (AP)(12,34),(56,78)Second line(90,12),(134,156)"
            ),
            [("ATLANTA (AP)", 12, 34, 56, 78), ("Second line", 90, 12, 134, 156)],
        )

    def test_incidental_parenthesis_then_more_text_is_kept(self):
        # Reported by @pspdada: an incidental parenthesis in the middle of an
        # item's text ("ATLANTA (AP) Second") followed by more text before the
        # coordinates used to split the item at "(AP)" and drop "ATLANTA (AP)",
        # leaving only ("Second", 1, 2, 3, 4).
        self.assertEqual(
            spotting._parse_format3("ATLANTA (AP) Second(1,2),(3,4)"),
            [("ATLANTA (AP) Second", 1, 2, 3, 4)],
        )

    def test_cjk_parenthetical_annotation_is_kept(self):
        self.assertEqual(
            spotting._parse_format3("金额(小写)(10,20),(30,40)"),
            [("金额(小写)", 10, 20, 30, 40)],
        )

    def test_extra_trailing_coordinate_pairs_still_bind_to_the_first_pair(self):
        self.assertEqual(
            spotting._parse_format3("Hello(1,2),(3,4),(5,6),(7,8)"),
            [("Hello", 1, 2, 3, 4)],
        )

    def test_multiline_text_is_still_accepted(self):
        self.assertEqual(
            spotting._parse_format3("line1\nline2(1,2),(3,4)"),
            [("line1\nline2", 1, 2, 3, 4)],
        )


class TestSpottingReward(unittest.TestCase):
    REF = "ATLANTA (AP)(12,34),(56,78)Second line(90,12),(134,156)"

    def _reward(self, response):
        return spotting.process_spotting_task(response, self.REF)["reward"]

    def test_detecting_a_parenthesised_region_beats_omitting_it(self):
        # Before the fix the parenthesised region was invisible to the scorer,
        # so omitting it scored a perfect 1.0 while detecting it scored lower --
        # the reward ordering was inverted.
        detected = self._reward("ATLANTA AP(12,34),(56,78)Second line(90,12),(134,156)")
        omitted = self._reward("Second line(90,12),(134,156)")
        self.assertGreater(detected, omitted)
        self.assertLess(omitted, 1.0)


if __name__ == "__main__":
    unittest.main()
