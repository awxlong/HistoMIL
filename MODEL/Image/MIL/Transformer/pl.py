import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import wandb
from matplotlib import pyplot as plt

from paras import get_loss, get_optimizer, get_scheduler
from HistoMIL.MODEL.Image.MIL.Transformer.paras import TransformerParas
from HistoMIL.MODEL.Image.MIL.Transformer.model import Transformer

DEFAULT_PARAS = TransformerParas(input_dim=1024, pretrained_weights='MSI_high_CRC_model.pth', encoder_name='pre-calculated')
    

class ClassifierLightning(pl.LightningModule):
    def __init__(self, paras:TransformerParas = DEFAULT_PARAS):
        super().__init__()
        self.config = paras
        self.model = Transformer(paras)
        self.criterion = get_loss(paras.criterion, pos_weight=paras.pos_weight
                                 ) if paras.task == "binary" else get_loss(paras.criterion)
        self.save_hyperparameters()

        self.lr = paras.lr
        self.wd = paras.wd

        self.acc_train = torchmetrics.Accuracy(task=paras.task, num_classes=paras.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=paras.task, num_classes=paras.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=paras.task, num_classes=paras.num_classes)

        self.auroc_val = torchmetrics.AUROC(
            task=paras.task,
            num_classes=paras.num_classes,
        )
        self.auroc_test = torchmetrics.AUROC(
            task=paras.task,
            num_classes=paras.num_classes,
        )

        self.f1_val = torchmetrics.F1Score(
            task=paras.task,
            num_classes=paras.num_classes,
        )
        self.f1_test = torchmetrics.F1Score(
            task=paras.task,
            num_classes=paras.num_classes,
        )

        self.precision_val = torchmetrics.Precision(
            task=paras.task,
            num_classes=paras.num_classes,
        )
        self.precision_test = torchmetrics.Precision(
            task=paras.task,
            num_classes=paras.num_classes,
        )

        self.recall_val = torchmetrics.Recall(
            task=paras.task,
            num_classes=paras.num_classes,
        )
        self.recall_test = torchmetrics.Recall(
            task=paras.task,
            num_classes=paras.num_classes,
        )

        self.specificity_val = torchmetrics.Specificity(
            task=paras.task,
            num_classes=paras.num_classes,
        )
        self.specificity_test = torchmetrics.Specificity(
            task=paras.task,
            num_classes=paras.num_classes,
        )

        self.cm_val = torchmetrics.ConfusionMatrix(task=paras.task, num_classes=paras.num_classes)
        self.cm_test = torchmetrics.ConfusionMatrix(
            task=paras.task, num_classes=paras.num_classes
        )

    def forward(self, x, *args):
        logits = self.model(x, *args)
        return logits

    def configure_optimizers(self, ):
        optimizer = get_optimizer(
            name=self.paras.optimizer,
            model=self.model,
            lr=self.lr,
            wd=self.wd,
        )
        if self.config.lr_scheduler:
            scheduler = get_scheduler(
                self.config.lr_scheduler,
                optimizer,
                **self.config.lr_scheduler_config,
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        x, coords, y, _, _ = batch  # x = features, coords, y = labels, tiles, patient
        logits = self.forward(x, coords)
        if self.config.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(0).float())
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1, keepdim=True)

        if self.config.task == "binary":
            self.acc_train(preds, y.unsqueeze(1))
        else:
            probs = torch.softmax(logits, dim=1)
            self.acc_train(probs, y)
        self.log("acc/train", self.acc_train, prog_bar=True)
        self.log("loss/train", loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, coords, y, _, _ = batch  # x = features, coords, y = labels, tiles, patient
        logits = self.forward(x, coords)
        if self.config.task == "binary":
            y = y.unsqueeze(1)
            loss = self.criterion(logits, y.float())
            probs = torch.sigmoid(logits)
        else:
            loss = self.criterion(logits, y)
            probs = torch.softmax(logits, dim=1)

        self.acc_val(probs, y)
        self.auroc_val(probs, y)
        self.f1_val(probs, y)
        self.precision_val(probs, y)
        self.recall_val(probs, y)
        self.specificity_val(probs, y)
        self.cm_val(probs, y)

        self.log("loss/val", loss, prog_bar=True)
        self.log("acc/val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc/val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1/val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision/val", self.precision_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall/val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "specificity/val", self.specificity_val, prog_bar=False, on_step=False, on_epoch=True
        )

    def on_validation_epoch_end(self):
        if self.global_step != 0:
            cm = self.cm_val.compute()

            # normalise the confusion matrix
            norm = cm.sum(axis=1, keepdims=True)
            normalized_cm = cm / norm

            # log to wandb
            plt.clf()
            cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
            wandb.log({"confusion_matrix/val": wandb.Image(cm)})

        self.cm_val.reset()

    def on_test_epoch_start(self) -> None:
        # save test outputs in dataframe per test dataset
        column_names = ['patient', 'ground_truth', 'predictions', 'logits', 'correct']
        self.outputs = pd.DataFrame(columns=column_names)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, coords, y, _, patient = batch  # x = features, coords, y = labels, tiles, patient
        logits = self.forward(x, coords)

        if self.config.task == "binary":
            y = y.unsqueeze(1)
            loss = self.criterion(logits, y.float())
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)
        else:
            loss = self.criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True)

        self.acc_test(probs, y)
        self.auroc_test(probs, y)
        self.f1_test(probs, y)
        self.precision_test(probs, y)
        self.recall_test(probs, y)
        self.specificity_test(probs, y)
        self.cm_test(probs, y)

        self.log("loss/test", loss, prog_bar=False)
        self.log("acc/test", self.acc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc/test", self.auroc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1/test", self.f1_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "precision/test", self.precision_test, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log("recall/test", self.recall_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "specificity/test", self.specificity_test, prog_bar=False, on_step=False, on_epoch=True
        )

        outputs = pd.DataFrame(
            data=[
                [patient[0],
                 y.item(),
                 preds.item(),
                 logits.squeeze(), (y == preds).int().item()]
            ],
            columns=['patient', 'ground_truth', 'prediction', 'logits', 'correct']
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)

    def on_test_epoch_end(self):
        if self.global_step != 0:
            cm = self.cm_test.compute()

            # normalise the confusion matrix
            norm = cm.sum(axis=1, keepdims=True)
            normalized_cm = cm / norm

            # log to wandb
            plt.clf()
            cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
            wandb.log({"confusion_matrix/test": wandb.Image(cm)})

        self.cm_test.reset()
