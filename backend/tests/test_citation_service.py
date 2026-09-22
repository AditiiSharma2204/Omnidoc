from app.services.citation_service import CitationService


class TestExtractCitedIndices:

    def test_single_citation(self):
        answer = "Revenue grew significantly [1]."
        assert CitationService.extract_cited_indices(answer) == {1}

    def test_multiple_distinct_citations(self):
        answer = "Revenue grew [1]. Costs also rose [2]."
        assert CitationService.extract_cited_indices(answer) == {1, 2}

    def test_adjacent_citations_on_one_claim(self):
        answer = "Both documents agree on this [1][3]."
        assert CitationService.extract_cited_indices(answer) == {1, 3}

    def test_repeated_citation_counted_once(self):
        answer = "Fact one [1]. Fact two [1]."
        assert CitationService.extract_cited_indices(answer) == {1}

    def test_no_citations_returns_empty_set(self):
        answer = "I couldn't find that information in the uploaded documents."
        assert CitationService.extract_cited_indices(answer) == set()

    def test_non_numeric_brackets_are_ignored(self):
        answer = "This is unsupported [citation needed]."
        assert CitationService.extract_cited_indices(answer) == set()

    def test_multi_digit_index(self):
        answer = "See the twelfth source [12]."
        assert CitationService.extract_cited_indices(answer) == {12}
