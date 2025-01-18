'''
Pytorch Lightning wrapper for Transformer
'''
from typing import override
import torch
from HistoMIL.MODEL.Image.Pl_protocol.BaseMIL import BaseMIL

import pdb


class pl_Transformer(BaseMIL):
    @override
    def forward(self, x):
        logits = self.model(x)
        return logits

    @override
    def infer_step(self, batch):
        self.eval()
        with torch.no_grad():
            x, y = batch
            logits, Y_prob, Y_hat, A_raw = self.model.infer(x)
        return logits, Y_prob, Y_hat, A_raw
