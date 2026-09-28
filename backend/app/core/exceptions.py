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


class DocumentNotFoundException(HTTPException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=404,
            detail=f"Document '{document_id}' not found."
        )


class ConversationNotFoundException(HTTPException):
    def __init__(self, conversation_id: str):
        super().__init__(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found."
        )