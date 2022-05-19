from typing import Optional
import torch.nn.functional as F
import pandas as pd
import tqdm
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

# 生成节点属性
token2int = {x:i for i, x in enumerate('GAVLIPFYWSTCMNQDEKRHXZ')}
onehot = F.one_hot(torch.tensor([i for i in token2int.values()])).numpy()
token2onehot = {x:oh for x, oh in zip(list('GAVLIPFYWSTCMNQDEKRHXZ'), onehot)}
int2token = {v: k for k,v in token2int.items()}

def onehot2token(tensors):
    return ''.join([int2token[i] for i in torch.argmax(tensors, axis=-1).tolist()])

def get_X(peptides: str):
    return torch.tensor([token2onehot[x] for x in peptides], dtype=torch.float)

# 生成边索引
def get_edge_index(peptides: str):
    l = len(peptides.rstrip('X'))
    a = [i for i in range(0, l-1)]
    a.append(l-1)
    b = [j for j in range(1, l)]
    b.append(0)
    return torch.tensor([a, b], dtype=torch.long)  # 索引的数值类型必须 long

def make_data(df):
    # 构建数据集，转换为张量
    dataset = []
    len_df = len(df)

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

        # Transform into Tensor
        self.train_data = make_data(self._train_data)
        self.val_data = make_data(self._val_data)
        self.test_data = make_data(self._test_data)


    def train_dataloader(self):
        return DataLoader(self.train_data, self.hparams.batch_size, num_workers=12, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, self.hparams.batch_size, shuffle=False, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.hparams.batch_size, shuffle=False, num_workers=12)













