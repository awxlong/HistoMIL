"""
pytorch-lightning wrapper for the model
"""

#---->
import pytorch_lightning as pl
import pandas as pd
import pytorch_lightning as pl

import seaborn as sns
import torch
import torchmetrics
import wandb
from matplotlib import pyplot as plt
#---->
from HistoMIL import logger
# from HistoMIL.MODEL.Image.PL_protocol.MIL import pl_MIL
# from HistoMIL.EXP.paras.dataset import DatasetParas
# from HistoMIL.EXP.paras.optloss import OptLossParas
# from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.MODEL.Image.MIL.TransMILRegression.paras import TransMILRegressionParas
from HistoMIL.MODEL.Image.MIL.TransMILRegression.model import TransMILRegression
from HistoMIL.MODEL.Image.MIL.utils import  get_loss, get_optimizer, get_scheduler
import pdb
#---->
####################################################################################
#      pl protocol class
####################################################################################

class pl_TransMILRegression(pl.LightningModule):
    def __init__(self, paras:TransMILRegressionParas):
        super().__init__()
        self.paras = paras
        self.model = TransMILRegression(paras)
        self.criterion = get_loss(paras.criterion
                                 ) if paras.task == "binary" else get_loss(paras.criterion)
        self.save_hyperparameters()

        self.lr = paras.lr
        self.wd = paras.wd

        ### Regression metrics
        ## MSE
        self.mse_train = torchmetrics.MeanSquaredError()
        self.mse_val = torchmetrics.MeanSquaredError()
        self.mse_test = torchmetrics.MeanSquaredError()
        ## PCC
        self.pearson_corr_train = torchmetrics.PearsonCorrCoef()
        self.pearson_corr_val = torchmetrics.PearsonCorrCoef()
        self.pearson_corr_test = torchmetrics.PearsonCorrCoef()

        ### Classification metrics (for binarized preds and labels)
        self.acc_train = torchmetrics.Accuracy(task='binary', num_classes=paras.num_classes)
        self.acc_val = torchmetrics.Accuracy(task='binary', num_classes=paras.num_classes)
        self.acc_test = torchmetrics.Accuracy(task='binary', num_classes=paras.num_classes)

        self.auroc_val = torchmetrics.AUROC(
            task='binary',
            num_classes=paras.num_classes,
        )
        self.auroc_test = torchmetrics.AUROC(
            task='binary',
            num_classes=paras.num_classes,
        )

        self.f1_val = torchmetrics.F1Score(
            task='binary',
            num_classes=paras.num_classes,
        )
        self.f1_test = torchmetrics.F1Score(
            task='binary',
            num_classes=paras.num_classes,
        )

        self.precision_val = torchmetrics.Precision(
            task='binary',
            num_classes=paras.num_classes,
        )
        self.precision_test = torchmetrics.Precision(
            task='binary',
            num_classes=paras.num_classes,
        )

        self.recall_val = torchmetrics.Recall(
            task='binary',
            num_classes=paras.num_classes,
        )
        self.recall_test = torchmetrics.Recall(
            task='binary',
            num_classes=paras.num_classes,
        )

        self.specificity_val = torchmetrics.Specificity(
            task='binary',
            num_classes=paras.num_classes,
        )
        self.specificity_test = torchmetrics.Specificity(
            task='binary',
            num_classes=paras.num_classes,
        )

        self.cm_val = torchmetrics.ConfusionMatrix(task='binary', num_classes=paras.num_classes)
        self.cm_test = torchmetrics.ConfusionMatrix(
            task='binary', num_classes=paras.num_classes
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

    def configure_optimizers(self, ):
        optimizer = get_optimizer(
            name=self.paras.optimizer,
            model=self.model,
            lr=self.lr,
            wd=self.wd,
        )
        if self.paras.lr_scheduler:
            scheduler = get_scheduler(
                self.paras.lr_scheduler,
                optimizer,
                self.paras.lr_scheduler_config,
            )
            # pdb.set_trace()
            return [optimizer], [scheduler]
        else:
            return [optimizer]
    def binarize(self, x):
        return (x <= self.paras.threshold).float()
    
    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        x, _, _, y = batch  # x = features, clinical_features, adj_matrix, y = regression scores
        
        logits = self.forward(x)
        # pdb.set_trace()
        ### regression logs
        loss = self.criterion(logits, y.unsqueeze(0).float()) # mse loss
        self.pearson_corr_train(logits, y.unsqueeze(0).float())
        self.log("pearson_corr_train", self.pearson_corr_train, prog_bar=True, on_step=False, on_epoch=True)

        # self.mse_train(logits, y)
        # self.log("mse_train", self.mse_train, prog_bar=True) # redundant
        self.log("mse_loss_train", loss, prog_bar=False)

        binary_preds = self.binarize(logits)
        binary_targets = self.binarize(y)
        binary_targets = binary_targets.unsqueeze(0)

        self.acc_train(binary_preds, binary_targets)
        self.log("acc_train", self.acc_train, prog_bar=True)

    
        # self.log("loss_train", loss, prog_bar=False)
        # pdb.set_trace()
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, _, _, y = batch  # x = features, clinical_features, adj_matrix, y = regression scores
        
        logits = self.forward(x)
        
        ## regression logs
        loss = self.criterion(logits, y.unsqueeze(0).float())
        self.pearson_corr_val(logits, y.unsqueeze(0).float())

        # pdb.set_trace()
        # self.log("loss_val", loss, prog_bar=True)
        # self.log("mse_val", self.mse_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_val", loss, prog_bar=True)
        self.log("pearson_corr_val", self.pearson_corr_val, prog_bar=True, on_step=False, on_epoch=True)

        # print(y)
        # print(y.shape)
        binary_preds = self.binarize(logits)
        binary_targets = self.binarize(y)
        binary_targets = binary_targets.unsqueeze(0) 

        self.acc_val(binary_preds, binary_targets)
        self.auroc_val(logits, binary_targets)
        self.f1_val(binary_preds, binary_targets)
        self.precision_val(binary_preds, binary_targets)
        self.recall_val(binary_preds, binary_targets)
        self.specificity_val(binary_preds, binary_targets)
        self.cm_val(binary_preds, binary_targets)

        self.log("acc_val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc_val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1_val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_val", self.precision_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall_val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("specificity_val", self.specificity_val, prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if self.global_step != 0:
            cm = self.cm_val.compute()

            # normalise the confusion matrix
            norm = cm.sum(axis=1, keepdims=True)
            normalized_cm = cm / norm

            # log to wandb
            plt.clf()
            cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
            wandb.log({"confusion_matrix_val": wandb.Image(cm)})

        self.cm_val.reset()

    def on_test_epoch_start(self) -> None:
        # save test outputs in dataframe per test dataset
        column_names = ['patient', 'ground_truth', 'predictions', 'logits', 'correct']
        self.outputs = pd.DataFrame(columns=column_names)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, _, y = batch  # x = features, clinical_features, adj_matrix, y = regression scores
        
        # pdb.set_trace()
        logits = self.forward(x)
        # regression logs
        loss = self.criterion(logits, y.unsqueeze(0).float())
        self.pearson_corr_test(logits, y.unsqueeze(0).float())

        self.log("loss_test", loss, prog_bar=False)
        self.log("pearson_corr_test", self.pearson_corr_test, prog_bar=True, on_step=False, on_epoch=True)

        binary_preds = self.binarize(logits)
        binary_targets = self.binarize(y)
        binary_targets = binary_targets.unsqueeze(0)
        
        self.acc_test(binary_preds, binary_targets)
        self.auroc_test(logits, binary_targets)
        self.f1_test(binary_preds, binary_targets)
        self.precision_test(binary_preds, binary_targets)
        self.recall_test(binary_preds, binary_targets)
        self.specificity_test(binary_preds, binary_targets)
        self.cm_test(binary_preds, binary_targets)
        
        self.log("acc_test", self.acc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc_test", self.auroc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1_test", self.f1_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_test", self.precision_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall_test", self.recall_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log("specificity_test", self.specificity_test, prog_bar=False, on_step=False, on_epoch=True)

        outputs = pd.DataFrame(
            data=[[y.item(), binary_preds.item(), logits.squeeze().item(), (binary_targets == binary_preds).int().item()]],
            columns=['ground_truth', 'prediction', 'logits', 'correct']
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
            wandb.log({"confusion_matrix_test": wandb.Image(cm)})

        self.cm_test.reset()

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()
