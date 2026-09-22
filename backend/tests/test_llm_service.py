import requests

import pytest

from app.services.llm_service import LLMService


class _FakeResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # Don't actually wait through backoff in tests.
    import app.services.llm_service as llm

    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


class TestGenerateRetries:

    def test_succeeds_on_first_try(self, monkeypatch):
        calls = []

        def fake_post(url, json, timeout):
            calls.append(1)
            return _FakeResponse(
                200, {"message": {"content": "hello"}}
            )

        monkeypatch.setattr(requests, "post", fake_post)

        result = LLMService.generate(system="sys", user="usr")

        assert result == "hello"
        assert len(calls) == 1

    def test_retries_on_500_then_succeeds(self, monkeypatch):
        responses = [
            _FakeResponse(500),
            _FakeResponse(500),
            _FakeResponse(200, {"message": {"content": "ok now"}}),
        ]
        calls = []

        def fake_post(url, json, timeout):
            calls.append(1)
            return responses[len(calls) - 1]

        monkeypatch.setattr(requests, "post", fake_post)

        result = LLMService.generate(system="sys", user="usr")

        assert result == "ok now"
        assert len(calls) == 3

    def test_raises_after_exhausting_retries(self, monkeypatch):
        def fake_post(url, json, timeout):
            return _FakeResponse(500)

        monkeypatch.setattr(requests, "post", fake_post)

        with pytest.raises(requests.exceptions.HTTPError):
            LLMService.generate(system="sys", user="usr")

    def test_does_not_retry_on_4xx(self, monkeypatch):
        calls = []

        def fake_post(url, json, timeout):
            calls.append(1)
            return _FakeResponse(400)

        monkeypatch.setattr(requests, "post", fake_post)

        with pytest.raises(requests.exceptions.HTTPError):
            LLMService.generate(system="sys", user="usr")

        # Must fail fast on a client error, not burn through retries.
        assert len(calls) == 1

    def test_retries_on_connection_error(self, monkeypatch):
        calls = []

        def fake_post(url, json, timeout):
            calls.append(1)
            if len(calls) < 2:
                raise requests.exceptions.ConnectionError(
                    "refused"
                )
            return _FakeResponse(200, {"message": {"content": "back up"}})

        monkeypatch.setattr(requests, "post", fake_post)

        result = LLMService.generate(system="sys", user="usr")

        assert result == "back up"
        assert len(calls) == 2
