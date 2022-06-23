import os
import random

from pandas import DataFrame

from .components.smiles_tokenizer import SmilesTokenizer

from typing import Optional, List, Tuple
import pandas as pd
from rich.progress import track
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pickle

from src import utils
from .components.rdkit_tools import pep2smi

log = utils.get_logger(__name__)


class PepSmiDataSet(Dataset):
    def __init__(self, peps, labels):
        super().__init__()
        self.peps = peps
        self.labels = labels

    def __len__(self):  # 样本数
        return len(self.peps)

    def __getitem__(self, idx):
        return self.peps[idx], self.labels[idx]


def make_data(df, tokenizer, data_type: Optional[str] = None):
    if data_type:
        log.info(f'processing {data_type} data')
    else:
        log.info('processing data')

    cached = f'/tmp/cache_pepsmi_{data_type}.pkl'
    if os.path.exists(cached):
        log.info(f'load data_from {cached}')
        with open(cached, 'rb') as f:
            dataset = pickle.load(f)
    else:
        # 构建数据集，转换为张量
        pps = []
        labels = []
        # for i in tqdm.trange(len_df):
        for i in track(range(len(df)), description=f"Processing...(total is {len(df)})"):
            pps.append(
                tokenizer.add_padding_tokens(tokenizer.encode(pep2smi(df.peptide[i])),
                                             length=260))
            labels.append(df.label[i])
        dataset = PepSmiDataSet(torch.LongTensor(pps), torch.LongTensor(labels))
        with open(cached, 'wb') as f:
            pickle.dump(dataset, f)
            log.info(f'dump dataset in to {cached}')
    return dataset


class Peptide_smilesModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.tokenizer = SmilesTokenizer(self.hparams.vocab_path)

    def prepare_data(self) -> None:

        # Load from csv
        self._train_data = pd.read_csv(self.hparams.train_data_path, index_col=0)
        self._val_data = pd.read_csv(self.hparams.val_data_path, index_col=0)
        self._test_data = pd.read_csv(self.hparams.test_data_path, index_col=0)

        if self.hparams.toy_data:
            self._train_data = self._train_data[:10000]
            self._val_data = self._val_data[:1000]
            self._test_data = self._test_data[:1000]

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = make_data(self._train_data, self.tokenizer, 'fit_train')
            self.val_data = make_data(self._val_data, self.tokenizer, 'fit_val')
            self.test_data = make_data(self._test_data, self.tokenizer, 'fit_test')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = make_data(self._test_data, self.tokenizer, 'test')

        if stage == "predict" or stage is None:
            self.predict_data = make_data(self._predict_data, self.tokenizer, "predict")

    def train_dataloader(self):
        # sampler 不能和 shuffle 搭配
        # return DataLoader(self.train_data, self.hparams.batch_size, num_workers=12, sampler=self.get_sampler())
        return DataLoader(self.train_data, self.hparams.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_data, self.hparams.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.hparams.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=32)


class Peptide_smilesModuleV2(Peptide_smilesModule):
    """V2
    1. 去重
    """

    def __init__(self, *args, **kwargs):
        super(Peptide_smilesModuleV2, self).__init__(*args, **kwargs)

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.tokenizer = SmilesTokenizer(self.hparams.vocab_path)

    def prepare_data(self) -> None:
        # Load from csv
        self._train_data = pd.read_csv(self.hparams.train_data_path, index_col=0).drop_duplicates(
            subset=['peptide']).reset_index(drop=True)
        self._val_data = pd.read_csv(self.hparams.val_data_path, index_col=0).drop_duplicates(
            subset=['peptide']).reset_index(drop=True)
        self._test_data = pd.read_csv(self.hparams.test_data_path, index_col=0).drop_duplicates(
            subset=['peptide']).reset_index(drop=True)

        if self.hparams.toy_data:
            self._train_data = self._train_data[:10000]
            self._val_data = self._val_data[:1000]
            self._test_data = self._test_data[:1000]


class Peptide_smilesModuleV0(Peptide_smilesModule):
    """V0
    1. 去重
    2. 测试用的假数据
    """

    def prepare_data(self) -> None:
        # 编造虚假数据
        # hydra 的数据路径没有用了

        token2int = {x: i for i, x in enumerate('GAVLIPFYWSTCMNQDEKRH')}

        def get_sentence() -> str:
            """生成一条多肽序列
            总长度为 9 到 12
            """
            return ''.join([random.choice('GAVLIPFYWSTCMNQDEKRH') for i in range(random.randint(9, 12))])

        def get_label(sentence: str) -> int:
            sum = 0
            for token in list(sentence):
                sum += token2int[token]
            return int(sum > 100)

        def get_sentences_with_label(n: int) -> Tuple[List[str], List[int]]:
            """生成 n 条多肽"""
            sentences = []
            labels = []
            for i in range(n):
                sentence = get_sentence()
                sentences.append(sentence)
                labels.append((get_label(sentence)))
            return sentences, labels

        pps, ys = get_sentences_with_label(10000)
        df = pd.DataFrame({'peptide': pps, 'label': ys}).drop_duplicates(subset=['peptide']).reset_index(drop=True)
        delta = int(len(df) * 0.1)
        self._test_data = df.iloc[:delta].reset_index(drop=True)
        self._val_data = df.iloc[delta:delta*2].reset_index(drop=True)
        self._train_data = df.iloc[delta*2:].reset_index(drop=True)



