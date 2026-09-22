from pathlib import Path

# Supported file extensions
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".docx",
    ".pptx",
    ".xlsx",
}

# Maximum upload size (25 MB)
MAX_FILE_SIZE = 25 * 1024 * 1024

# Supported MIME types
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "image/png",
    "image/jpeg",
}

# Extensions Docling can convert directly (routed to DoclingParser)
DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}

# Storage directories
UPLOAD_DIR = Path("storage/uploads")