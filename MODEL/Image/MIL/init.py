from HistoMIL.MODEL.Image.MIL.ABMIL.model import ABMIL
from HistoMIL.MODEL.Image.MIL.TransMIL.model import TransMIL
from HistoMIL.MODEL.Image.MIL.DSMIL.model import DSMIL
from HistoMIL.MODEL.Image.MIL.Transformer.model import Transformer
from HistoMIL.MODEL.Image.MIL.AttentionMIL.model import AttentionMIL
from HistoMIL.MODEL.Image.MIL.CAMIL.model import CAMIL
from HistoMIL.MODEL.Image.MIL.DTFD_MIL.model import DTFD_MIL



aviliable_mil_models = {
                    "ABMIL":ABMIL,

                    "TransMIL":TransMIL,   

                    "DSMIL":DSMIL,

                    "Transformer": Transformer,

                    'AttentionMIL': AttentionMIL,

                    'CAMIL': CAMIL,

                    'DTFD-MIL': DTFD_MIL,
    }