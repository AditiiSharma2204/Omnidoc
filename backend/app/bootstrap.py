"""
Process-startup fixes that must run before ANYTHING else is
imported. Any entry point that imports docling or sentence
transformers (the FastAPI app, the eval harness, a one-off script)
must call `apply_local_first_env_fixes()` as its very first
statement -- huggingface_hub reads these env vars once at import
time, so setting them any later than that is too late.

Kept separate from app/main.py so non-server entry points (eval
scripts, notebooks) get the same fixes without importing FastAPI.
"""
import os


def apply_local_first_env_fixes() -> None:
    # Defensive fix for a broken local conda env: SSL_CERT_FILE can
    # end up pointing at a cacert.pem that doesn't exist (seen on
    # this project's dev machine). httpx/requests then crash
    # constructing an SSL context for ANY HTTPS request -- before
    # offline checks even get a chance to run. Repoint it at
    # certifi's bundle instead of leaving the whole app hostage to
    # one bad env var.
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    if not ssl_cert_file or not os.path.isfile(ssl_cert_file):
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            os.environ.pop("SSL_CERT_FILE", None)

    # OmniDoc is local-first: once models are cached, nothing should
    # ever make a network call to HuggingFace Hub just to check for
    # updates.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
