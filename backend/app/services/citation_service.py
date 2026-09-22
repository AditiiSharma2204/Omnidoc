import re

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CitationService:
    """
    Parses [N]-style inline citation markers out of a generated
    answer. Pure text processing -- no model calls, no state.
    """

    @staticmethod
    def extract_cited_indices(answer: str) -> set[int]:
        """
        Returns the set of distinct context numbers the model cited,
        e.g. {1, 3} for an answer containing "...revenue grew [1][3].
        Costs also rose [1]." Malformed markers (non-digit brackets
        like "[citation needed]") simply don't match and are ignored.
        """
        return {int(m) for m in _CITATION_PATTERN.findall(answer)}
