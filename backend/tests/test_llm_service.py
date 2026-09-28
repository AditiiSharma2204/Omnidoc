import requests

import pytest

from app.services.llm_service import LLMService


class _FakeResponse:
    def __init__(self, status_code, json_data=None, lines=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._lines = lines or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._json_data

    def iter_lines(self):
        return iter(self._lines)


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


def _ndjson_line(token: str) -> bytes:
    import json as _json

    return _json.dumps({"message": {"content": token}}).encode()


class TestStreamRetries:
    """
    LLMService.stream() previously had no retry logic at all, unlike
    generate() -- this hit the documented Ollama/CUDA crash live (a
    real chat request right after BGE-M3+reranker loaded got a 500
    from Ollama's own CUDA init failing) with nothing to recover it.
    Mirrors TestGenerateRetries: retries apply to establishing the
    connection only, not to token iteration itself.
    """

    def test_succeeds_on_first_try(self, monkeypatch):
        calls = []

        def fake_post(url, json, stream, timeout):
            calls.append(1)
            return _FakeResponse(
                200, lines=[_ndjson_line("hel"), _ndjson_line("lo")]
            )

        monkeypatch.setattr(requests, "post", fake_post)

        tokens = list(LLMService.stream(system="sys", user="usr"))

        assert tokens == ["hel", "lo"]
        assert len(calls) == 1

    def test_retries_on_500_then_succeeds(self, monkeypatch):
        responses = [
            _FakeResponse(500),
            _FakeResponse(500),
            _FakeResponse(200, lines=[_ndjson_line("ok now")]),
        ]
        calls = []

        def fake_post(url, json, stream, timeout):
            calls.append(1)
            return responses[len(calls) - 1]

        monkeypatch.setattr(requests, "post", fake_post)

        tokens = list(LLMService.stream(system="sys", user="usr"))

        assert tokens == ["ok now"]
        assert len(calls) == 3

    def test_raises_after_exhausting_retries(self, monkeypatch):
        def fake_post(url, json, stream, timeout):
            return _FakeResponse(500)

        monkeypatch.setattr(requests, "post", fake_post)

        with pytest.raises(requests.exceptions.HTTPError):
            list(LLMService.stream(system="sys", user="usr"))

    def test_does_not_retry_on_4xx(self, monkeypatch):
        calls = []

        def fake_post(url, json, stream, timeout):
            calls.append(1)
            return _FakeResponse(400)

        monkeypatch.setattr(requests, "post", fake_post)

        with pytest.raises(requests.exceptions.HTTPError):
            list(LLMService.stream(system="sys", user="usr"))

        assert len(calls) == 1

    def test_retries_on_connection_error(self, monkeypatch):
        calls = []

        def fake_post(url, json, stream, timeout):
            calls.append(1)
            if len(calls) < 2:
                raise requests.exceptions.ConnectionError("refused")
            return _FakeResponse(200, lines=[_ndjson_line("back up")])

        monkeypatch.setattr(requests, "post", fake_post)

        tokens = list(LLMService.stream(system="sys", user="usr"))

        assert tokens == ["back up"]
        assert len(calls) == 2


class TestMessageHistory:

    def test_no_history_sends_just_system_and_user(self):
        messages = LLMService._messages(system="sys", user="usr")

        assert messages == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]

    def test_history_inserted_between_system_and_new_user_turn(self):
        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]

        messages = LLMService._messages(
            system="sys", user="new question", history=history
        )

        assert messages == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "new question"},
        ]

    def test_generate_passes_history_through_to_ollama(self, monkeypatch):
        captured = {}

        def fake_post(url, json, timeout):
            captured["messages"] = json["messages"]
            return _FakeResponse(200, {"message": {"content": "ok"}})

        monkeypatch.setattr(requests, "post", fake_post)

        LLMService.generate(
            system="sys",
            user="new question",
            history=[{"role": "user", "content": "prior"}],
        )

        assert captured["messages"][1] == {
            "role": "user",
            "content": "prior",
        }
