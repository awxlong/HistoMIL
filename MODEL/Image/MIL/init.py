from HistoMIL.MODEL.Image.MIL.ABMIL.model import ABMIL
from HistoMIL.MODEL.Image.MIL.TransMIL.model import TransMIL
from HistoMIL.MODEL.Image.MIL.DSMIL.model import DSMIL
from HistoMIL.MODEL.Image.MIL.Transformer.model import Transformer

aviliable_mil_models = {
                    "ABMIL":ABMIL,

                    "TransMIL":TransMIL,   

                    "DSMIL":DSMIL,

                    "Transformer": Transformer
    }