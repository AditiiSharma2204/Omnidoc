from pathlib import Path

from fastapi import UploadFile

from app.core.constants import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE,
)

from app.core.exceptions import (
    InvalidFileTypeException,
    FileTooLargeException,
    EmptyFileException,
)


class ValidationService:

    @staticmethod
    async def validate(file: UploadFile):

        extension = Path(file.filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise InvalidFileTypeException()

        if file.content_type not in ALLOWED_MIME_TYPES:
            raise InvalidFileTypeException()

        content = await file.read()

        size = len(content)

        if size == 0:
            raise EmptyFileException()

        if size > MAX_FILE_SIZE:
            raise FileTooLargeException()

        file.file.seek(0)

        return size