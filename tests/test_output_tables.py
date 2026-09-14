import unittest

from inference.utils.output_utils import normalize_doc_parse_markdown


class TableCoordinatesTest(unittest.TestCase):
    def test_multiline_table_preserves_coordinates_and_cleans_following_prose(self):
        table = "<table>\n<tr>\n<td>\n(2,4),(3,3)\n</td>\n</tr>\n</table>"
        actual, stats = normalize_doc_parse_markdown(table + "\nFigure(1,2),(3,4)")
        self.assertEqual(actual, table + "\nFigure")
        self.assertEqual(stats, {"U_coord_text": 1})

    def test_uppercase_table_and_header_cells_are_preserved(self):
        table = (
            '<TABLE class="points">\n<TR><TH>\nPoints(2,4),(3,3)\n</TH></TR>\n</TABLE>'
        )
        actual, stats = normalize_doc_parse_markdown(table)
        self.assertEqual(actual, table)
        self.assertEqual(stats, {})

    def test_nested_table_does_not_end_outer_table_protection(self):
        table = "<table>\n<tr><td><table><tr><td>nested</td></tr></table>\n(2,4),(3,3)\n</td></tr>\n</table>"
        actual, stats = normalize_doc_parse_markdown(table + "\n(1,2),(3,4)")
        self.assertEqual(actual, table)
        self.assertEqual(stats, {"U_coord_bare": 1})

    def test_table_opening_tag_can_span_lines(self):
        table = '<table\nclass="points">\n<tr><td>\n(2,4),(3,3)\n</td></tr>\n</table>'
        actual, stats = normalize_doc_parse_markdown(table)
        self.assertEqual(actual, table)
        self.assertEqual(stats, {})

    def test_math_coordinates_and_plain_coordinate_cleanup(self):
        actual, stats = normalize_doc_parse_markdown("$(2,4),(3,3)$\n(1,2),(3,4)")
        self.assertEqual(actual, "$(2,4),(3,3)$")
        self.assertEqual(stats, {"U_coord_bare": 1})


if __name__ == "__main__":
    unittest.main()
