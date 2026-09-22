import json
import time

import requests

from app.config.settings import settings


class LLMService:
    """
    Service responsible for interacting with the local Ollama server.
    """

    # Ollama's llama-server subprocess can crash on its OWN CUDA
    # initialization when a PyTorch process (e.g. this app's embedding/
    # reranker models) is concurrently resident on this machine's
    # Windows/WDDM + MX450 setup -- diagnosed directly, not guessed:
    # the error is "llama-server process has terminated ... CUDA error:
    # shared object initialization failed", it reproduces even with
    # CUDA hidden from our own process (CUDA_VISIBLE_DEVICES=""), even
    # with only ~2GB resident and >3GB system RAM free (so it is NOT
    # primarily a memory-quantity problem), and even with only ONE
    # torch model loaded (not specific to running both the embedder
    # and reranker). It looks like a genuine driver-level race, not
    # something this app can prevent -- but Ollama auto-respawns
    # llama-server, and a 20s wait (measured directly) reliably lets
    # that respawn succeed while the torch process stays resident.
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 20

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
    def unload(cls) -> bool:
        """
        Best-effort release of Ollama's resident model.

        Ollama keeps a loaded model resident (system RAM + whatever
        fits in VRAM) for a keep-alive window after the last call,
        independent of this app's own process. On memory-constrained
        hardware, that resident model competing with this process
        loading the embedding model is enough to OOM either process.
        Callers that are about to do heavy local memory work (e.g.
        the eval harness loading BGE-M3) can call this first to free
        that headroom; Ollama reloads transparently (with a one-time
        reload delay) on the next real request.

        Returns True if the request succeeded, False otherwise --
        never raises, since this is a courtesy call, not something
        that should fail the caller's real work.
        """
        try:
            requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={"model": settings.LLM_MODEL_NAME, "keep_alive": 0},
                timeout=30,
            )
            return True
        except requests.RequestException:
            return False

    @classmethod
    def generate(
        cls,
        system: str,
        user: str,
        temperature: float = None,
    ) -> str:

        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": cls._messages(system, user),
            "options": cls._options(
                settings.LLM_TEMPERATURE
                if temperature is None
                else temperature
            ),
            "stream": False,
        }

        last_error = None

        for attempt in range(cls.MAX_RETRIES):

            try:
                response = requests.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()

                data = response.json()

                return data["message"]["content"].strip()

            except requests.exceptions.HTTPError as e:
                # Only retry server-side (5xx) errors -- a 4xx means
                # our request is wrong, and retrying it won't help.
                status = e.response.status_code if e.response else None
                if status is None or status < 500:
                    raise
                last_error = e

            except requests.exceptions.ConnectionError as e:
                # Ollama can be mid-restart/reload and refuse the
                # connection outright, not just return a 5xx.
                last_error = e

            if attempt < cls.MAX_RETRIES - 1:
                time.sleep(cls.RETRY_BACKOFF_SECONDS * (attempt + 1))

        raise last_error

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
