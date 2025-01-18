"""
Pytorch-Lightning wrapper for AttentionMIL 
"""
from typing import override

from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.MODEL.Image.MIL.AttentionMIL.paras import AttentionMILParas, DEFAULT_Attention_MIL_PARAS
from HistoMIL.MODEL.Image.MIL.AttentionMIL.model import AttentionMIL
from HistoMIL.MODEL.Image.MIL.utils import  get_optimizer, get_scheduler
from HistoMIL.MODEL.Image.Pl_protocol.BaseMIL import BaseMIL

import pdb

class pl_AttentionMIL(BaseMIL):
    def __init__(self, dataset_paras: DatasetParas, paras: AttentionMILParas = DEFAULT_Attention_MIL_PARAS):
        """
        Initialize the AttentionMIL model.

        Args:
            dataset_paras (DatasetParas): Dataset-specific parameters.
            paras (AttentionMILParas): Model-specific parameters.
        """
        # Store dataset_paras for use in configure_optimizers
        self.dataset_paras = dataset_paras

        # Call the BaseMIL constructor with only model_paras
        super().__init__(model_paras=paras)

    def _create_model(self):
        """
        Create and return the AttentionMIL model.
        """
        return AttentionMIL(self.paras)

    @override
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        Override this method to use dataset_paras for steps_per_epoch
        """
        optimizer = get_optimizer(
            name=self.paras.optimizer,
            model=self.model,
            lr=self.lr,
            wd=self.wd,
        )
        if self.paras.lr_scheduler:
            self.paras.lr_scheduler_config = {
                'pct_start': 0.25,  # Default 0.3, set same as HistoBistro
                'epochs': self.paras.epoch,
                'max_lr': self.paras.max_lr,
                'steps_per_epoch': self.dataset_paras.data_len  # Use dataset_paras for steps_per_epoch
            }
            scheduler = get_scheduler(
                self.paras.lr_scheduler,
                optimizer,
                self.paras.lr_scheduler_config,
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]
    
