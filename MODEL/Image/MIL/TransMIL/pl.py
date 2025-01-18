"""
pytorch-lightning wrapper for TransMIL
"""

from HistoMIL.MODEL.Image.Pl_protocol.BaseMIL import BaseMIL
from HistoMIL.MODEL.Image.MIL.TransMIL.model import TransMIL

class pl_TransMIL(BaseMIL):
    def _create_model(self):
        """Create and return the TransMIL model."""
        return TransMIL(self.paras)