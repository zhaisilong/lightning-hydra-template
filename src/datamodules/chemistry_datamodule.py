import os
from functools import partial

from src.utils.mol_tokenizers import (
    SimpleTokenizer, MolTokenizer, AtomTokenizer, SelfiesTokenizer
)
from src.utils.chemistry_data import TaskPrefixDataset, data_collator, LineByLineTextDataset
from transformers import T5ForConditionalGeneration
from typing import Dict, Optional
import pytorch_lightning as pl
from transformers import BatchEncoding, PreTrainedTokenizer

class ProductDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            dataset: Optional[LineByLineTextDataset, TaskPrefixDataset],
            type
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab/simple.pt')
        self.tokenizer = tokenizer(vocab_file=vocab_path)

    def prepare_data(self) -> None:
        dataset = TaskPrefixDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.data_dir,
            prefix=self.hparams.prefix,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
            separate_vocab=self.hparams.output_layer,
            type_path="train",
        )
        data_collator_padded = partial(
            data_collator, pad_token_id=self.tokenizer.pad_token_id)
        eval_strategy = "steps"

    def train_dataloader(self):
        return DataLoader()

        eval_iter = TaskPrefixDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.data_dir,
            prefix=task.prefix,
            max_source_length=task.max_source_length,
            max_target_length=task.max_target_length,
            separate_vocab=(task.output_layer != 'seq2seq'),
            type_path="val",
        )
