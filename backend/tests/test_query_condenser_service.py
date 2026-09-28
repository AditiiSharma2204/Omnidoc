from app.services.query_condenser_service import QueryCondenserService


class TestQueryCondenserService:

    def test_no_history_returns_question_unchanged(self, monkeypatch):
        import app.services.query_condenser_service as qcs

        def fail_generate(**kwargs):
            raise AssertionError("should not call the LLM with no history")

        monkeypatch.setattr(qcs.LLMService, "generate", fail_generate)

        result = QueryCondenserService.condense("What is her name?", [])

        assert result == "What is her name?"

    def test_condenses_using_history(self, monkeypatch):
        import app.services.query_condenser_service as qcs

        captured = {}

        def fake_generate(system, user, temperature=None):
            captured["system"] = system
            captured["user"] = user
            return "What was Aditii Sharma's second job?"

        monkeypatch.setattr(qcs.LLMService, "generate", fake_generate)

        history = [
            {"role": "user", "content": "What is this person's name?"},
            {
                "role": "assistant",
                "content": "The person's name is Aditii Sharma.",
            },
        ]

        result = QueryCondenserService.condense(
            "what about her second job?", history
        )

        assert result == "What was Aditii Sharma's second job?"
        assert "Aditii Sharma" in captured["user"]
        assert "what about her second job?" in captured["user"]

    def test_blank_llm_output_falls_back_to_original_question(
        self, monkeypatch
    ):
        import app.services.query_condenser_service as qcs

        monkeypatch.setattr(
            qcs.LLMService, "generate", lambda **kwargs: "   "
        )

        history = [{"role": "user", "content": "prior turn"}]

        result = QueryCondenserService.condense("follow-up?", history)

        assert result == "follow-up?"
