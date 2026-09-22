import json

import requests

from app.config.settings import settings


class LLMService:
    """
    Service responsible for interacting with the local Ollama server.
    """

    @classmethod
    def _options(cls, temperature: float) -> dict:
        return {
            "temperature": temperature,
            # Without an explicit num_ctx, Ollama falls back to a
            # small default context window and silently truncates
            # anything past it -- including, potentially, the
            # question itself. PromptBuilder budgets its context to
            # this same value, so the two must move together.
            "num_ctx": settings.LLM_NUM_CTX,
        }

    @classmethod
    def _messages(cls, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @classmethod
    def generate(
        cls,
        system: str,
        user: str,
        temperature: float = None,
    ) -> str:

        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": settings.LLM_MODEL_NAME,
                "messages": cls._messages(system, user),
                "options": cls._options(
                    settings.LLM_TEMPERATURE
                    if temperature is None
                    else temperature
                ),
                "stream": False,
            },
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"].strip()

    @classmethod
    def stream(
        cls,
        system: str,
        user: str,
        temperature: float = None,
    ):
        """
        Stream tokens from Ollama.
        """

        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": settings.LLM_MODEL_NAME,
                "messages": cls._messages(system, user),
                "options": cls._options(
                    settings.LLM_TEMPERATURE
                    if temperature is None
                    else temperature
                ),
                "stream": True,
            },
            stream=True,
            timeout=300,
        )

        response.raise_for_status()

        for line in response.iter_lines():

            if not line:
                continue

            data = json.loads(line)

            if "message" in data:
                token = data["message"].get("content", "")

                if token:
                    yield token
