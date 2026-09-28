# The ML stack (torch, docling, sentence-transformers, faiss) makes
# this a large image no matter what -- expected and accepted for a
# local-first app whose whole point is running these models on your
# own machine, not something a multi-stage build meaningfully shrinks.
FROM python:3.11-slim

WORKDIR /app

# System libraries Docling's PDF/image pipeline and OpenCV-backed
# deps need at runtime; kept to the minimum found necessary rather
# than installing a full build toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Overridden by docker-compose.yml for the real container paths --
# these are just sane defaults if the image is run standalone.
ENV DOCUMENTS_DIR=storage/documents \
    VECTORSTORE_DIR=storage/vectorstore \
    DATABASE_PATH=storage/app.db

RUN mkdir -p storage/documents storage/vectorstore

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
