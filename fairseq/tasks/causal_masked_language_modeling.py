# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    MonolingualDataset,
    TokenBlockDataset,
    StripTokenDataset,
    IncludeTargetsForCausalLanguageModeling,
    SafeDataset,
    PrependTokenDataset,
    NestedDictionaryDataset,
    AppendTokenDataset,
    IdDataset,
    PadDataset,
    NumelDataset,
    AssertDataset,
    data_utils,
)
from fairseq.data.causal_masked_dataset import CausalMaskedDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask
logger = logging.getLogger(__name__)

SENTINEL_SELECTION = ChoiceEnum(["fixed", "poisson"])


@dataclass
class HTLMCausallyMaskedConfig(LanguageModelingConfig):
    num_sentinel_tokens: int = field(
        default=512,
        metadata={"help": ""},
    )
    num_sentinel_tokens: int = field(
        default=1,
        metadata={"help": ""},
    )
    sentinel_method: SENTINEL_SELECTION = field(
        default="fixed",
        metadata={
            "help": "Whether or not to dynamically sample number of sentinel tokens (masks) to be placed per document"
        },
    )


@register_task("causal_masked_language_modeling", dataclass=HTLMCausallyMaskedConfig)
class CausalMaskedLanguageModelingTask(LanguageModelingTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)
        assert output_dictionary is None or dictionary == output_dictionary
        self.sentinel_token_expectation = args.num_sentinel_tokens
        self.sentinel_tokens = []
        self.sentinel_method = str(args.sentinel_method)
        for i in range(256 if self.sentinel_method != "fixed" else args.num_sentinel_tokens):
            self.sentinel_tokens.append(
                self.dictionary.add_symbol(f"<sentinel:{i}>"))
        assert self.targets == [
            "future"], "HTLM Causally Masked only works with future targets"

        logger.info(
            f"Setting criterion weights with {len(self.sentinel_tokens)} Sentinel Tokens but with expectation of {self.sentinel_token_expectation} and policy of {self.sentinel_method}.")

        self.sentinel_end_token = self.dictionary.add_symbol("<eoss>")
        self.criterion_weights = torch.ones(len(self.dictionary))
        for i in range(len(self.sentinel_tokens)):
            self.criterion_weights[self.sentinel_tokens[i]] = 0.0

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        with torch.no_grad():
            paths = utils.split_paths(self.args.data)
            assert len(paths) > 0

            data_path = paths[(epoch - 1) % len(paths)]
            split_path = os.path.join(data_path, split)

            # each process has its own copy of the raw data (likely to be an np.memmap)
            dataset = data_utils.load_indexed_dataset(
                split_path, self.dictionary, self.args.dataset_impl, combine=combine
            )
            if dataset is None:
                raise FileNotFoundError(
                    f"Dataset not found: {split} ({split_path})")

            assert self.args.sample_break_mode == "eos" or "eos_blocked" == self.args.sample_break_mode, "Every item should be one document"
            dataset = StripTokenDataset(dataset, self.dictionary.eos())
            dataset = StripTokenDataset(dataset, self.dictionary.index("▁"))

            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode=self.args.sample_break_mode,
                include_targets=False,
                use_plasma_view=self.args.use_plasma_view,
                split_path=split_path,
                plasma_path=self.args.plasma_path,
            )
            assert "eos_blocked" == self.args.sample_break_mode, self.args.sample_break_mode

            dataset = CausalMaskedDataset(dataset, self.sentinel_token_expectation,
                                          self.sentinel_tokens, self.sentinel_method,
                                          self.args.tokens_per_sample, self.sentinel_end_token)
            dataset = IncludeTargetsForCausalLanguageModeling(
                dataset, self.source_dictionary)

            add_eos_for_other_targets = (
                self.args.sample_break_mode is not None
                and self.args.sample_break_mode != "none"
            )
            fixed_pad_length = None
            if self.args.pad_to_fixed_length:
                fixed_pad_length = self.args.tokens_per_sample

            pad_to_bsz = None
            if self.args.pad_to_fixed_bsz:
                pad_to_bsz = self.args.batch_size_valid if 'valid' in split else self.args.batch_size

            dataset = SafeDataset(MonolingualDataset(
                dataset=dataset,
                sizes=dataset.sizes,
                src_vocab=self.dictionary,
                tgt_vocab=self.output_dictionary,
                add_eos_for_other_targets=add_eos_for_other_targets,
                shuffle=True,
                targets=self.targets,
                add_bos_token=self.args.add_bos_token,
                fixed_pad_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            ), safe_collate=True, rounds_index_selection=5)

            dataset = AssertDataset(dataset)
            dataset.add_assertion(lambda x: x['source'].size(0) < self.args.tokens_per_sample,
                                  "Dataset produces more tokens than allowed in source")
            dataset.add_assertion(lambda x: x['target'].size(0) < self.args.tokens_per_sample,
                                  "Dataset produces more tokens than allowed in target")
            dataset.add_assertion(lambda x: (x['source'] < len(self.source_dictionary)).all(),
                                  "Dataset produces tokens outside vocab in source")
            dataset.add_assertion(lambda x: (x['target'] < len(self.target_dictionary)).all(),
                                  "Dataset produces tokens outside vocab in target")

            self.datasets[split] = dataset

            logger.info(
                "Split: {0}, Loaded {1} samples of htlm_causal".format(
                    split,
                    len(self.datasets[split]),
                )
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        with torch.no_grad():
            dataset = StripTokenDataset(
                TokenBlockDataset(
                    src_tokens,
                    src_lengths,
                    block_size=None,  # ignored for "eos" break mode
                    pad=self.source_dictionary.pad(),
                    eos=self.source_dictionary.eos(),
                    break_mode="eos",
                ),
                # remove eos from (end of) target sequence
                self.source_dictionary.eos(),
            )
            dataset = StripTokenDataset(dataset, self.dictionary.index("▁"))
            src_dataset = PrependTokenDataset(
                dataset,
                token=(
                    self.source_dictionary.bos()
                    if getattr(self.args, "add_bos_token", False)
                    else self.source_dictionary.eos()
                ),
            )
            tgt_dataset = AppendTokenDataset(
                dataset, token=self.source_dictionary.pad())
            return NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "src_tokens": PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "target": PadDataset(
                        tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                    ),
                },
                sizes=[np.array(src_lengths)],
            )
