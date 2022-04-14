from functools import partial

from src.datamodules.components.datasets import ReverseDataset
import pytorch_lightning as pl
from typing import Optional, Tuple
from torch.utils import data

class ReverseDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[data.Dataset] = None
        self.data_val: Optional[data.Dataset] = None
        self.data_test: Optional[data.Dataset] = None

    def prepare_data(self):
        self.dataset = partial(ReverseDataset, 10, 16)

    def train_dataloader(self):
        return data.DataLoader(self.dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return data.DataLoader(self.dataset(1000), batch_size=128)

    def test_dataloader(self):
        return data.DataLoader(self.dataset(10000), batch_size=128)
