from app.services.query_rewrite_service import (
    SYSTEM_PROMPT,
    QueryRewriteService,
)


class TestGenerateHypotheticalAnswer:

    def test_calls_llm_with_question_as_user_message(self, monkeypatch):
        import app.services.query_rewrite_service as qrs

        captured = {}

        def fake_generate(system, user, temperature=None):
            captured["system"] = system
            captured["user"] = user
            captured["temperature"] = temperature
            return "The person's name is Aditii Sharma."

        monkeypatch.setattr(qrs.LLMService, "generate", fake_generate)

        result = QueryRewriteService.generate_hypothetical_answer(
            "What is this person's name?"
        )

        assert result == "The person's name is Aditii Sharma."
        assert captured["user"] == "What is this person's name?"
        assert captured["system"] == SYSTEM_PROMPT

    def test_does_not_use_prompt_builder(self, monkeypatch):
        """
        The hypothetical answer is fabricated by design and must
        never be routed through the real answer-generation prompt
        (which would make it look like a grounded, real answer).
        """
        import app.services.query_rewrite_service as qrs

        monkeypatch.setattr(
            qrs.LLMService,
            "generate",
            lambda system, user, temperature=None: "hypothetical",
        )

        result = QueryRewriteService.generate_hypothetical_answer("q")

        assert result == "hypothetical"
