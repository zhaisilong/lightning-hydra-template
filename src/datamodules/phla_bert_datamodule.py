from typing import Optional
import torch.nn.functional as F
import pandas as pd
from rich.progress import track
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np

from src import utils

log = utils.get_logger(__name__)

# 生成节点属性
token2int = {x: i for i, x in enumerate('GAVLIPFYWSTCMNQDEKRHX')}
onehot = F.one_hot(torch.tensor([i for i in token2int.values()])).numpy()
token2onehot = {x: oh for x, oh in zip(list('GAVLIPFYWSTCMNQDEKRHX'), onehot)}
int2token = {v: k for k, v in token2int.items()}


def onehot2token(tensors):
    return ''.join([int2token[i] for i in torch.argmax(tensors, axis=-1).tolist()])


def get_X(peptides: str):
    return torch.tensor(np.array([token2onehot[x] for x in peptides]), dtype=torch.float)


# 生成边索引
def get_edge_index(peptides: str):
    """一维序列生成边索引
    """
    length = len(peptides)
    a = [i for i in range(0, length - 1)]
    b = [j for j in range(1, length)]
    return torch.tensor([a, b], dtype=torch.long)  # 索引的数值类型必须 long

def get_edge_index_with_sep(peptides: str):
    """
    ABCXEDF -> [[0,1,4,5], [1,2,5,6]]
    """
    part_a, part_b = peptides.split('X')
    len_a, len_b = len(part_a), len(part_b)
    _from = [a_i for a_i in range(0, len_a - 1)] + [a_j for a_j in range(len_a + 1, len_a + len_b)]
    _to = [i + 1 for i in _from]
    return torch.tensor(np.array([_from, _to]), dtype=torch.long)


def make_data(df, data_type: Optional[str] = None):
    # 构建数据集，转换为张量
    dataset = []
    len_df = len(df)

    if data_type:
        log.info(f'processing {data_type} data')
    else:
        log.info('processing data')


    # for i in tqdm.trange(len_df):
    for i in track(range(len_df), description="Processing..."):
        hla_peptide = df.peptide[i] + 'X' + df.HLA_sequence[i]
        dataset.append(Data(
            x=get_X(hla_peptide),  # 编码特征使用 多肽 + hla
            # x=get_X(df.peptide[i]),  # 只编码多肽
            # edge_index=get_edge_index(df.peptide[i]),  # 边特征只用多肽
            edge_index=get_edge_index_with_sep(hla_peptide),
            y=torch.tensor(df.label[i], dtype=torch.long),
            pp=hla_peptide))  # pp 用于记录对应的 item

    return dataset


class PeptideModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:

        # Load from csv
        self._train_data = pd.read_csv(self.hparams.train_data_path, index_col=0)
        self._val_data = pd.read_csv(self.hparams.val_data_path, index_col=0)
        self._test_data = pd.read_csv(self.hparams.test_data_path, index_col=0)
        self._predict_data = pd.read_csv(self.hparams.test_data_path, index_col=0)

        if self.hparams.toy_data:
            self._train_data = self._train_data[:100000]
            self._val_data = self._val_data[:1000]
            self._test_data = self._test_data[:1000]
            self._predict_data = self._predict_data[:1000]

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = make_data(self._train_data, 'fit/train')
            self.val_data = make_data(self._val_data, 'fit/val')
            self.test_data = make_data(self._test_data, 'fit/test')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = make_data(self._test_data, 'test')

        if stage == "predict" or stage is None:
            self.predict_data = make_data(self._predict_data, "predict")

    def train_dataloader(self):
        # sampler 不能和 shuffle 搭配
        # return DataLoader(self.train_data, self.hparams.batch_size, num_workers=12, sampler=self.get_sampler())
        return DataLoader(self.train_data, self.hparams.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.test_data, self.hparams.batch_size, shuffle=True, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.hparams.batch_size, shuffle=False, num_workers=6)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=32)
