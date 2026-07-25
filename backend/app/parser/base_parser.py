from abc import ABC, abstractmethod


class BaseParser(ABC):

    @abstractmethod
    def parse(self, file_path: str):
        """
        Parse a document and return structured content.
        """
        pass