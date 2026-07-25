from fastapi import HTTPException


class InvalidFileTypeException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=400,
            detail="Unsupported file type."
        )


class FileTooLargeException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=400,
            detail="File exceeds maximum allowed size."
        )


class EmptyFileException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=400,
            detail="Uploaded file is empty."
        )