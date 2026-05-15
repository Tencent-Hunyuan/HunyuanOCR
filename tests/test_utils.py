import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "Hunyuan-OCR-master" / "utils.py"

spec = importlib.util.spec_from_file_location("hyocr_utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class SpottingResponseParserTest(unittest.TestCase):
    def test_spotting_text_can_contain_parentheses(self):
        response = (
            "ATLANTA (AP) - Human(43,500),(325,512)"
            "they (detectives)(371,523),(653,534)"
        )

        items = list(utils._iter_spotting_items(response))

        self.assertEqual(
            [item[0] for item in items],
            ["ATLANTA (AP) - Human", "they (detectives)"],
        )

    def test_process_spotting_response_preserves_parenthesized_text(self):
        response = (
            "ATLANTA (AP) - Human(43,500),(325,512)"
            "they (detectives)(371,523),(653,534)"
        )

        processed = utils.process_spotting_response(
            response,
            image_width=2000,
            image_height=1000,
        )

        self.assertIn("ATLANTA (AP) - Human(86,500),(650,512)", processed)
        self.assertIn("they (detectives)(742,523),(1306,534)", processed)


if __name__ == "__main__":
    unittest.main()
