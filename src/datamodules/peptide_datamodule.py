from typing import Optional
import torch.nn.functional as F
import pandas as pd
import tqdm
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

from src import utils

log = utils.get_logger(__name__)


# 生成节点属性
token2int = {x: i for i, x in enumerate('GAVLIPFYWSTCMNQDEKRHXZ')}
onehot = F.one_hot(torch.tensor([i for i in token2int.values()])).numpy()
token2onehot = {x: oh for x, oh in zip(list('GAVLIPFYWSTCMNQDEKRHXZ'), onehot)}
int2token = {v: k for k, v in token2int.items()}


def onehot2token(tensors):
    return ''.join([int2token[i] for i in torch.argmax(tensors, axis=-1).tolist()])


def get_X(peptides: str):
    return torch.tensor([token2onehot[x] for x in peptides], dtype=torch.float)


# 生成边索引
def get_edge_index(peptides: str):
    l = len(peptides.rstrip('X'))
    a = [i for i in range(0, l - 1)]
    a.append(l - 1)
    b = [j for j in range(1, l)]
    b.append(0)
    return torch.tensor([a, b], dtype=torch.long)  # 索引的数值类型必须 long


def make_data(df, data_type: Optional[str] = None):
    # 构建数据集，转换为张量
    dataset = []
    len_df = len(df)

    if data_type:
        log.info(f'processing {data_type} data')
    else:
        log.info('processing data')

    for i in tqdm.trange(len_df):
        paded_peptides = df.paded_sentence[i]
        dataset.append(Data(
            x=get_X(paded_peptides),
            edge_index=get_edge_index(paded_peptides),
            y=torch.tensor(df.label[i], dtype=torch.long),
            v=torch.tensor(df.enrich[i], dtype=torch.float),
            pp=df.sentence[i]))

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

        # self.class_sample_count = [
        #     len(self._train_data[self._train_data['label'] == 0]),
        #     len(self._train_data[self._train_data['label'] == 1])
        # ]

    # def get_sampler(self):
    #     # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    #     weights = 1 / torch.Tensor(self.class_sample_count)
    #     # 注意这里的 weights 应为所有样本的权重序列，其长度为所有样本长度。
    #     return torch.utils.data.sampler.WeightedRandomSampler(weights,
    #                                                           self.hparams.batch_size)

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
        return DataLoader(self.train_data, self.hparams.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.test_data, self.hparams.batch_size, shuffle=True, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.hparams.batch_size, shuffle=False, num_workers=12)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=32)
