"""
pytorch-lightning wrapper for TransMIL with multimodal fusion of clinical variables

"""

#---->
from typing import override
import pandas as pd

import torch
#---->
from HistoMIL import logger


from HistoMIL.MODEL.Image.MIL.TransMILMultimodal.paras import TransMILMultimodalParas
from HistoMIL.MODEL.Image.MIL.TransMILMultimodal.model import TransMILMultimodal
from HistoMIL.MODEL.Image.MIL.BaseMIL import BaseMIL


import pdb
class pl_TransMILMultimodal(BaseMIL):
    def __init__(self, paras: TransMILMultimodalParas):
        """
        Initialize the TransMILMultimodal model.

        Args:
            paras (TransMILMultimodalParas): Model-specific parameters.
        """
        super().__init__(model_paras=paras)

    def _create_model(self):
        """
        Create and return the TransMILMultimodal model.
        """
        return TransMILMultimodal(self.paras)

    @override
    def forward(self, x, clinical_feats):
        """
        Forward pass for TransMILMultimodal. Overrides BaseMIL.forward to handle multimodal inputs.
        """
        logits = self.model(x, clinical_feats)
        return logits

    @override
    def training_step(self, batch, batch_idx):
        """
        Training step for TransMILMultimodal. Overrides BaseMIL.training_step to handle multimodal inputs.
        """
        x, clinical_feats, _, y = batch  # Unpack batch
        logits = self.forward(x, clinical_feats)

        if self.paras.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(0).float())
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1, keepdim=True)

        if self.paras.task == "binary":
            self.acc_train(preds, y.unsqueeze(1))
        else:
            probs = torch.softmax(logits, dim=1)
            self.acc_train(probs, y)

        self.log("acc_train", self.acc_train, prog_bar=True)
        self.log("loss_train", loss, prog_bar=False)

        return loss

    @override
    def validation_step(self, batch, batch_idx):
        """
        Validation step for TransMILMultimodal. Overrides BaseMIL.validation_step to handle multimodal inputs.
        """
        x, clinical_feats, _, y = batch  # Unpack batch
        logits = self.forward(x, clinical_feats)

        if self.paras.task == "binary":
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

        self.log("loss_val", loss, prog_bar=True)
        self.log("acc_val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc_val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1_val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_val", self.precision_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall_val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("specificity_val", self.specificity_val, prog_bar=False, on_step=False, on_epoch=True)

    @override
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Test step for TransMILMultimodal. Overrides BaseMIL.test_step to handle multimodal inputs.
        """
        x, clinical_feats, _, y = batch  # Unpack batch
        logits = self.forward(x, clinical_feats)

        if self.paras.task == "binary":
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

        self.log("loss_test", loss, prog_bar=False)
        self.log("acc_test", self.acc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc_test", self.auroc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1_test", self.f1_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_test", self.precision_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall_test", self.recall_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log("specificity_test", self.specificity_test, prog_bar=False, on_step=False, on_epoch=True)

        # Save test outputs
        outputs = pd.DataFrame(
            data=[[y.item(), preds.item(), torch.sigmoid(logits.squeeze()).item(), (y == preds).int().item()]],
            columns=['ground_truth', 'prediction', 'probs', 'correct']
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)

    @override
    def infer_step(self, batch):
        """
        Inference step for TransMILMultimodal. Overrides BaseMIL.infer_step to handle multimodal inputs and return additional outputs.
        """
        self.eval()
        with torch.no_grad():
            x, clinical_feats, _, y = batch  # Unpack batch
            logits, Y_prob, Y_hat, A_raw, clinical_gradients = self.model.infer(x, clinical_feats)
        return logits, Y_prob, Y_hat, A_raw, clinical_gradients