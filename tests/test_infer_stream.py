import json
import unittest

import httpx
from openai import APIError, OpenAI

from inference.utils.output_utils import infer_stream


def chunk(content):
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]})
        + "\n\n"
    ).encode()


class EventStream(httpx.SyncByteStream):
    def __init__(self, events):
        self.events = events

    def __iter__(self):
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield event


class InferStreamTest(unittest.TestCase):
    def run_stream(self, events, expected=None, error=None):
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=EventStream(events),
        )
        with OpenAI(
            api_key="test",
            base_url="http://localhost/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda request: response)
            ),
        ) as client:
            try:
                if error is not None:
                    with self.assertRaises(error):
                        infer_stream(client, {"model": "test", "messages": []})
                else:
                    self.assertEqual(
                        infer_stream(client, {"model": "test", "messages": []}),
                        expected,
                    )
                self.assertTrue(
                    response.is_closed,
                    "response must be closed before returning or propagating errors",
                )
            finally:
                response.close()

    def test_api_error_closes_response(self):
        self.run_stream(
            [chunk("partial"), b'data: {"error":{"message":"generation failed"}}\n\n'],
            error=APIError,
        )

    def test_interruption_closes_response(self):
        self.run_stream(
            [chunk("partial"), KeyboardInterrupt()], error=KeyboardInterrupt
        )

    def test_completed_response(self):
        self.run_stream(
            [chunk("recognized text"), b"data: [DONE]\n\n"],
            expected=("recognized text", False),
        )

    def test_repetition_stops_and_closes_response(self):
        self.run_stream(
            [chunk("ab" * 2000), chunk("not consumed")], expected=("ab" * 2000, True)
        )


if __name__ == "__main__":
    unittest.main()
