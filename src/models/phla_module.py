from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.auroc import AUROC
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PHLALitModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_auroc = AUROC(num_classes=2)
        self.test_auroc = AUROC(num_classes=2)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, pep: torch.Tensor, hla: torch.Tensor):
        return self.net(pep, hla)

    def step(self, batch: Any):
        pep, hla, y = batch
        logits, _, _, attns = self.forward(pep, hla)  # [b, 2]

        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y, logits

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.step(batch)

        probs = F.softmax(logits)

        # log val metrics
        acc = self.val_acc(preds, targets)
        auroc = self.val_auroc(probs, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        val_acc_best = self.val_acc_best.compute()

        self.log("val/acc_best", val_acc_best, on_step=False, on_epoch=True, prog_bar=True)
        self.log("hp_metric", val_acc_best, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.step(batch)

        probs = F.softmax(logits)

        # log test metrics
        acc = self.test_acc(preds, targets)
        auc = self.test_auroc(probs, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()
        self.val_auroc.reset()
        self.test_auroc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        opt = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        LR = get_cosine_schedule_with_warmup(opt, 10, 100)

        return {'optimizer': opt, 'lr_scheduler': LR}


class PHLALitModule_smooth(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.9)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_auroc = AUROC(num_classes=2)
        self.test_auroc = AUROC(num_classes=2)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, pep: torch.Tensor, hla: torch.Tensor):
        return self.net(pep, hla)

    def step(self, batch: Any):
        pep, hla, y = batch
        logits, _, _, attns = self.forward(pep, hla)  # [b, 2]

        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y, logits

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.step(batch)

        probs = F.softmax(logits)

        # log val metrics
        acc = self.val_acc(preds, targets)
        auroc = self.val_auroc(probs, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        val_acc_best = self.val_acc_best.compute()

        self.log("val/acc_best", val_acc_best, on_step=False, on_epoch=True, prog_bar=True)
        self.log("hp_metric", val_acc_best, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.step(batch)

        probs = F.softmax(logits)

        # log test metrics
        acc = self.test_acc(preds, targets)
        auc = self.test_auroc(probs, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()
        self.val_auroc.reset()
        self.test_auroc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        opt = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        LR = get_cosine_schedule_with_warmup(opt, 8, 80)

        return {'optimizer': opt, 'lr_scheduler': LR}
