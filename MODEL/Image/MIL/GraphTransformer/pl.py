"""
pytorch-lightning wrapper for the Graph Transformer
"""

from typing import override
import pandas as pd
import torch

from HistoMIL.MODEL.Image.MIL.GraphTransformer.paras import GraphTransformerParas, DEFAULT_GRAPHTRANSFORMER_PARAS
from HistoMIL.MODEL.Image.MIL.GraphTransformer.model import GraphTransformer
from HistoMIL.MODEL.Image.Pl_protocol.BaseMIL import BaseMIL

import pdb


class pl_GraphTransformer(BaseMIL):
    def __init__(self, paras: GraphTransformerParas = DEFAULT_GRAPHTRANSFORMER_PARAS):
        """
        Initialize the GraphTransformer model.

        Args:
            paras (GraphTransformerParas): Model-specific parameters.
        """
        super().__init__(model_paras=paras)

    def _create_model(self):
        """
        Create and return the GraphTransformer model.
        """
        return GraphTransformer(self.paras)

    @override
    def forward(self, x, sparse_adj_matrix):
        """
        Forward pass for GraphTransformer. Overrides BaseMIL.forward to handle sparse adjacency matrices.
        """
        logits, mc1, o1 = self.model(x, sparse_adj_matrix)
        return logits, mc1, o1

    @override
    def training_step(self, batch, batch_idx):
        """
        Training step for GraphTransformer. Overrides BaseMIL.training_step to handle custom loss components.
        """
        x, sparse_adj_matrix, y = batch  # Unpack batch
        logits, mc1, o1 = self.forward(x, sparse_adj_matrix[0])

        if self.paras.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(0).float()) + mc1 + o1
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
        Validation step for GraphTransformer. Overrides BaseMIL.validation_step to handle custom loss components.
        """
        x, sparse_adj_matrix, y = batch  # Unpack batch
        logits, mc1, o1 = self.forward(x, sparse_adj_matrix[0])

        if self.paras.task == "binary":
            y = y.unsqueeze(1)
            loss = self.criterion(logits, y.float()) + mc1 + o1
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
        Test step for GraphTransformer. Overrides BaseMIL.test_step to handle custom loss components.
        """
        x, sparse_adj_matrix, y = batch  # Unpack batch
        logits, mc1, o1 = self.forward(x, sparse_adj_matrix[0])

        if self.paras.task == "binary":
            y = y.unsqueeze(1)
            loss = self.criterion(logits, y.float()) + mc1 + o1
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
        Inference step for GraphTransformer. Overrides BaseMIL.infer_step to handle sparse adjacency matrices.
        """
        self.eval()
        with torch.no_grad():
            x, adj_matrix, y = batch  # Unpack batch
            logits, Y_prob, Y_hat, A_raw = self.model.infer(x, adj_matrix[0])
        return logits, Y_prob, Y_hat, A_raw