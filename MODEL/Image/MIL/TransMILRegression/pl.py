"""
pytorch-lightning wrapper for TransMILRegression
"""
from typing import override
import pandas as pd

import torch
import torchmetrics


from HistoMIL.MODEL.Image.MIL.TransMILRegression.paras import TransMILRegressionParas
from HistoMIL.MODEL.Image.MIL.TransMILRegression.model import TransMILRegression
from HistoMIL.MODEL.Image.Pl_protocol.BaseMIL import BaseMIL

#---->
####################################################################################
#      pl protocol class
####################################################################################
class pl_TransMILRegression(BaseMIL):
    def __init__(self, paras: TransMILRegressionParas):
        """
        Initialize the TransMILRegression model.

        Args:
            paras (TransMILRegressionParas): Model-specific parameters.
        """
        super().__init__(model_paras=paras)

        ### Regression metrics
        self.mse_train = torchmetrics.MeanSquaredError()
        self.mse_val = torchmetrics.MeanSquaredError()
        self.mse_test = torchmetrics.MeanSquaredError()

        self.pearson_corr_train = torchmetrics.PearsonCorrCoef()
        self.pearson_corr_val = torchmetrics.PearsonCorrCoef()
        self.pearson_corr_test = torchmetrics.PearsonCorrCoef()

        ### Classification metrics (for binarized preds and labels)
        self.acc_train = torchmetrics.Accuracy(task='binary', num_classes=paras.num_classes)
        self.acc_val = torchmetrics.Accuracy(task='binary', num_classes=paras.num_classes)
        self.acc_test = torchmetrics.Accuracy(task='binary', num_classes=paras.num_classes)

        self.auroc_val = torchmetrics.AUROC(task='binary', num_classes=paras.num_classes)
        self.auroc_test = torchmetrics.AUROC(task='binary', num_classes=paras.num_classes)

        self.f1_val = torchmetrics.F1Score(task='binary', num_classes=paras.num_classes)
        self.f1_test = torchmetrics.F1Score(task='binary', num_classes=paras.num_classes)

        self.precision_val = torchmetrics.Precision(task='binary', num_classes=paras.num_classes)
        self.precision_test = torchmetrics.Precision(task='binary', num_classes=paras.num_classes)

        self.recall_val = torchmetrics.Recall(task='binary', num_classes=paras.num_classes)
        self.recall_test = torchmetrics.Recall(task='binary', num_classes=paras.num_classes)

        self.specificity_val = torchmetrics.Specificity(task='binary', num_classes=paras.num_classes)
        self.specificity_test = torchmetrics.Specificity(task='binary', num_classes=paras.num_classes)

        self.cm_val = torchmetrics.ConfusionMatrix(task='binary', num_classes=paras.num_classes)
        self.cm_test = torchmetrics.ConfusionMatrix(task='binary', num_classes=paras.num_classes)

    def _create_model(self):
        """
        Create and return the TransMILRegression model.
        """
        return TransMILRegression(self.paras)

    def binarize(self, x):
        """
        Binarize the input using the threshold specified in the parameters.
        """
        return (x <= self.paras.threshold).float()

    @override
    def training_step(self, batch, batch_idx):
        """
        Training step for TransMILRegression. Overrides BaseMIL.training_step to handle regression and binary classification metrics.
        """
        x, _, _, y = batch  # Unpack batch
        logits = self.forward(x)

        ### Regression logs
        loss = self.criterion(logits, y.unsqueeze(0).float())  # MSE loss
        self.pearson_corr_train(logits, y.unsqueeze(0).float())

        self.log("pearson_corr_train", self.pearson_corr_train, prog_bar=True, on_step=False, on_epoch=True)
        self.log("mse_loss_train", loss, prog_bar=False)

        ### Binary classification logs
        binary_preds = self.binarize(logits)
        binary_targets = self.binarize(y).unsqueeze(0)

        self.acc_train(binary_preds, binary_targets)
        self.log("acc_train", self.acc_train, prog_bar=True)

        return loss

    @override
    def validation_step(self, batch, batch_idx):
        """
        Validation step for TransMILRegression. Overrides BaseMIL.validation_step to handle regression and binary classification metrics.
        """
        x, _, _, y = batch  # Unpack batch
        logits = self.forward(x)

        ### Regression logs
        loss = self.criterion(logits, y.unsqueeze(0).float())
        self.pearson_corr_val(logits, y.unsqueeze(0).float())

        self.log("loss_val", loss, prog_bar=True)
        self.log("pearson_corr_val", self.pearson_corr_val, prog_bar=True, on_step=False, on_epoch=True)

        ### Binary classification logs
        binary_preds = self.binarize(logits)
        binary_targets = self.binarize(y).unsqueeze(0)

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

    @override
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Test step for TransMILRegression. Overrides BaseMIL.test_step to handle regression and binary classification metrics.
        """
        x, _, _, y = batch  # Unpack batch
        logits = self.forward(x)

        ### Regression logs
        loss = self.criterion(logits, y.unsqueeze(0).float())
        self.pearson_corr_test(logits, y.unsqueeze(0).float())

        self.log("loss_test", loss, prog_bar=False)
        self.log("pearson_corr_test", self.pearson_corr_test, prog_bar=True, on_step=False, on_epoch=True)

        ### Binary classification logs
        binary_preds = self.binarize(logits)
        binary_targets = self.binarize(y).unsqueeze(0)

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

        ### Save test outputs
        outputs = pd.DataFrame(
            data=[[binary_targets.item(), y.item(), logits.squeeze().item(), binary_preds.item(), (binary_targets == binary_preds).int().item()]],
            columns=['binary_ground_truth', 'continuous_ground_truth', 'prediction', 'binary_predictions', 'binary_correct']
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)

    @override
    def infer_step(self, batch):
        """
        Inference step for TransMILRegression. Overrides BaseMIL.infer_step to handle regression outputs.
        """
        self.eval()
        with torch.no_grad():
            x, _, _, y = batch  # Unpack batch
            logits, Y_prob, Y_hat, A_raw = self.model.infer(x)
        return logits, Y_prob, Y_hat, A_raw