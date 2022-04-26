
import numpy as np
import torch

from . import FairseqDataset, data_utils
from fairseq.data.dictionary import Dictionary
from typing import List, Tuple
from . import BaseWrapperDataset


def span_intersection(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    left_x, left_y = left
    right_x, right_y = right
    return max(left_x, right_x) < min(left_y, right_y)


class IncludeTargetsForCausalLanguageModeling(BaseWrapperDataset):
    def __init__(self, dataset: FairseqDataset, dictionary: Dictionary) -> None:
        super().__init__(dataset)
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()
        self.dictionary = dictionary

    def __getitem__(self, index):
        item = self.dataset[index]
        buffer = item
        e = len(item)

        source = torch.cat([item.new([self.eos]), buffer[0: e - 1]])
        past_target = torch.cat(
            [item.new([self.pad, self.eos]), buffer[0: e - 2]]
        )
        return source, item, past_target


class CausalMaskedDataset(BaseWrapperDataset):
    def __init__(self, dataset: FairseqDataset,
                 sentinel_token_expectation: int,
                 sentinel_tokens: List[int],
                 sentinel_method: str,
                 tokens_per_sample: int, sentinel_eos: int):
        super().__init__(dataset)
        self.sentinel_token_expectation = sentinel_token_expectation
        self.sentinel_tokens = sentinel_tokens
        self.sentinel_method = sentinel_method
        self.tokens_per_sample = tokens_per_sample
        self.eos = sentinel_eos
        assert self.sentinel_method == "fixed" or self.sentinel_method == "poisson"
        assert len(self.sentinel_tokens) >= 1
        assert self.tokens_per_sample > 1

        self.sentinel_fixed = self.sentinel_method == "fixed"

    def get_sentinel(self, i):
        return self.sentinel_tokens[i]

    def sentinel_masking(self, document: torch.Tensor, spans: List[Tuple[int, int]]):
        document_clone = document.clone()
        document_retrieve_mask = torch.ones_like(document_clone).to(torch.bool)

        for i, span in enumerate(spans):
            document_clone[span[0]] = self.get_sentinel(i)
            document_retrieve_mask[span[0] + 1:span[1]] = False

        return document_clone[document_retrieve_mask]

    def sentinel_targets(self, document: torch.Tensor, spans: List[Tuple[int, int]]):
        num_focused_tokens = sum(x[1] - x[0] for x in spans)
        num_spans = len(spans)
        target = torch.zeros(num_focused_tokens + 2 * num_spans).to(document)
        index = 0
        if self.sentinel_fixed:
            assert len(self.sentinel_tokens) == len(spans)
        else:
            assert len(self.sentinel_tokens) > len(spans)

        for i, span in enumerate(spans):
            target[index] = self.get_sentinel(i)
            index += 1
            size = span[1] - span[0]
            target[index: index + size] = document[span[0]:span[1]]
            target[index + size] = self.eos
            index = index + size + 1
        return target

    def get_spans_to_mask(self, document_length: int) -> List[Tuple[int, int]]:
        # Ok, we do not use a budget here but instead
        # our goal is to sample from ~ U[0,1] in the case of len(sentinel_tokens) = 1
        # If len(sentinel_tokens) > 1 we try to find len(sentinel_tokens) non intersecting spans
        len_sentinel_tokens = self.sentinel_token_expectation if self.sentinel_fixed else torch.poisson(
            torch.tensor([float(self.sentinel_token_expectation)])).clamp(1, len(self.sentinel_tokens) - 1).to(torch.int).item()

        if len_sentinel_tokens == 1:
            start, end = np.random.uniform(size=2)
            if end < start:
                start, end = end, start
            # round down
            start = int(start * document_length)
            # round up
            end = int(end * document_length + 0.5)
            if start == end:
                return None
            else:
                assert start < end
                return [(start, end)]

        # Let's implement the general case. We will create len(self.sentinel_tokens) ** 2 possible candidates
        # And we will filter one by one to insure no intersections. If we can't find anything then so be it.
        return_spans: List[Tuple[int, int]] = []
        candidate_spans: List[Tuple[int, int]] = [
            tuple(np.random.uniform(size=2)) for _ in range(len_sentinel_tokens ** 2)]
        candidate_spans = [(int(start * document_length), int(end *
                            document_length + 0.5)) for (start, end) in candidate_spans]
        candidate_spans = [(start, end) if start <= end else (
            end, start) for (start, end) in candidate_spans]
        while len(return_spans) < len_sentinel_tokens and len(candidate_spans) > 0:
            candidate_span = candidate_spans.pop()
            if not any(span_intersection(x, candidate_span) for x in return_spans):
                return_spans.append(candidate_span)
        return return_spans

    def get_ordered_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return sorted(spans, key=lambda x: x[0])

    def __getitem__(self, index):
        with data_utils.numpy_seed(index):
            item = self.dataset[index]
            assert len(item) > 0
            spans = self.get_spans_to_mask(len(item))
            if spans is None:
                return item
            else:
                spans = self.get_ordered_spans(spans)
                causal_source = self.sentinel_masking(item, spans)
                causal_masked = self.sentinel_targets(item, spans)

            return torch.cat([causal_source, causal_masked])[:self.tokens_per_sample]
