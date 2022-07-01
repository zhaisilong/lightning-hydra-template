from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from typing import Optional, NoReturn, Union, Tuple, List
import numpy as np
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from src import utils

log = utils.get_logger(__name__)

token2int = {x: i for i, x in enumerate('-GAVLIPFYWSTCMNQDEKRH')}


def pep2idx(pep: str) -> List[int]:
    return [token2int[p] for p in list(pep)]


class MyDataSet(Dataset):
    def __init__(self, pep_inputs, hla_inputs, labels):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels

    def __len__(self):  # 样本数
        return len(self.pep_inputs)

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx]


def make_data(data, vocab):
    pep_max_len = 15  # peptide; enc_input max sequence length
    hla_max_len = 34  # hla; dec_input(=dec_output) max sequence length

    pep_inputs, hla_inputs, labels = [], [], []
    for pep, hla, label in zip(data.peptide, data.HLA_sequence, data.label):
        pep, hla = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        hla_input = [[vocab[n] for n in hla]]
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
        labels.append(label)
    return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels)


def data_with_loader(mode='train', batch_size=1024):
    if mode == 'train':
        data = pd.read_csv('../data/AOMP/train_data_fold0.csv', index_col=0)
    if mode == 'val':
        data = pd.read_csv('../data/AOMP/val_data_fold0.csv', index_col=0)

    pep_inputs, hla_inputs, labels = make_data(data)
    loader = DataLoader(MyDataSet(pep_inputs, hla_inputs, labels), batch_size, shuffle=False, num_workers=8)

    return data, pep_inputs, hla_inputs, labels, loader


class MyDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> NoReturn:
        # Load from csv
        vocab = np.load(self.hparams.vocab_dict_path, allow_pickle=True).item()
        self._train_data = pd.read_csv(self.hparams.train_data_path, index_col=0)
        self._val_data = pd.read_csv(self.hparams.val_data_path, index_col=0)
        self._test_data = pd.read_csv(self.hparams.test_data_path, index_col=0)

        # Transform into Tensor
        self.train_data = make_data(self._train_data, vocab)
        self.val_data = make_data(self._val_data, vocab)
        self.test_data = make_data(self._test_data, vocab)

    def train_dataloader(self):
        return DataLoader(MyDataSet(*self.train_data), self.hparams.batch_size, shuffle=False, num_workers=8)

    def val_dataloader(self):
        return DataLoader(MyDataSet(*self.val_data), self.hparams.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(MyDataSet(*self.test_data), self.hparams.batch_size, shuffle=False, num_workers=8)


class SmilesDataModel(LightningDataModule):
    def __init__(
            self,
            train_data_path,
            val_data_path,
            test_data_path,
            tokenizer_path,
            max_len: int = 150,
            batch_size: int = 1024,
            *args,
            **kwargs
    ):
        super(SmilesDataModel, self).__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.tokenizer = Tokenizer.from_file(self.hparams.tokenizer_path)
        self.tokenizer.enable_padding(length=self.hparams.max_len)
        self.tokenizer.enable_truncation(max_length=self.hparams.max_len)

    def _make_data(self, df, mode='train') -> Tuple[Tensor, Tensor, Tensor]:
        hla_max_len = 34  # hla; dec_input(=dec_output) max sequence length

        pep_inputs, hla_inputs, labels = [], [], []
        log.info(f'prepare {mode} data')
        with tqdm(zip(df.smiles, df.HLA_sequence, df.label), total=len(df)) as pbar:
            for pep, hla, label in pbar:
                hla = hla.ljust(hla_max_len, '-')

                pep_input = [self.tokenizer.encode(pep).ids]
                hla_input = [[token2int[n] for n in hla]]
                pep_inputs.extend(pep_input)
                hla_inputs.extend(hla_input)
                labels.append(label)
            return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels)

    def prepare_data(self) -> NoReturn:
        # Load from csv
        self._train_data = pd.read_csv(self.hparams.train_data_path, index_col=0)
        self._val_data = pd.read_csv(self.hparams.val_data_path, index_col=0)
        self._test_data = pd.read_csv(self.hparams.test_data_path, index_col=0)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = self._make_data(self._train_data, 'train')
            self.val_data = self._make_data(self._val_data, 'val')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = self._make_data(self._test_data, 'test')

    def train_dataloader(self):
        return DataLoader(MyDataSet(*self.train_data), self.hparams.batch_size, shuffle=False, num_workers=8)

    def val_dataloader(self):
        return DataLoader(MyDataSet(*self.val_data), self.hparams.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(MyDataSet(*self.test_data), self.hparams.batch_size, shuffle=False, num_workers=8)
