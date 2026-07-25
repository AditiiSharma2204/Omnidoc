from pathlib import Path

# Supported file extensions
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".docx",
    ".pptx",
}

# Maximum upload size (25 MB)
MAX_FILE_SIZE = 25 * 1024 * 1024

# Supported MIME types
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "image/png",
    "image/jpeg",
}

# Storage directories
UPLOAD_DIR = Path("storage/uploads")