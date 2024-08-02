"""
init function for pl models

"""
from HistoMIL.EXP.paras.trainer import PLTrainerParas
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas

from HistoMIL.MODEL.Image.MIL.init import aviliable_mil_models
# from HistoMIL.MODEL.Image.PL_protocol.MIL import pl_MIL


import pdb

from typing import Dict, Any
# from HistoMIL.MODEL.Image.MIL import (
#     ABMIL, TransMIL, DSMIL, Transformer, AttentionMIL, CAMIL, DTFD_MIL, GraphTransformer, TransMILMultimodal
# )


def create_img_model(
    train_paras: PLTrainerParas,
    dataset_paras: DatasetParas,
    optloss_paras: OptLossParas,
    model_para: Dict[str, Any]
) -> Any:
    """
    Create and return an image processing model based on the specified parameters.

    Args:
        train_paras (PLTrainerParas): Training parameters.
        dataset_paras (DatasetParas): Dataset parameters.
        optloss_paras (OptLossParas): Optimization and loss parameters.
        model_para (Dict[str, Any]): Model-specific parameters.

    Returns:
        Any: The created model instance.

    Raises:
        ValueError: If the specified model name is not available.
    """
    model_name = train_paras.model_name

    model_classes = {
        "ABMIL": "HistoMIL.MODEL.Image.MIL.ABMIL.pl",
        
        "DSMIL": "HistoMIL.MODEL.Image.MIL.DSMIL.pl",
        
        "Transformer": "HistoMIL.MODEL.Image.MIL.Transformer.pl",
        "TransformerMultimodal": "HistoMIL.MODEL.Image.MIL.TransformerMultimodal.pl",
        "TransformerRegression": "HistoMIL.MODEL.Image.MIL.TransformerRegression.pl",

        "AttentionMIL": "HistoMIL.MODEL.Image.MIL.AttentionMIL.pl",
        
        "CAMIL": "HistoMIL.MODEL.Image.MIL.CAMIL.pl",
        
        "DTFD_MIL": "HistoMIL.MODEL.Image.MIL.DTFD_MIL.pl",
        
        "GraphTransformer": "HistoMIL.MODEL.Image.MIL.GraphTransformer.pl",

        "TransMIL": "HistoMIL.MODEL.Image.MIL.TransMIL.pl",
        "TransMILMultimodal": "HistoMIL.MODEL.Image.MIL.TransMILMultimodal.pl",
        "DTFDTransMIL": "HistoMIL.MODEL.Image.MIL.DTFDTransMIL.pl",
        "TransMILRegression": "HistoMIL.MODEL.Image.MIL.TransMILRegression.pl",

        "CLAM" : "HistoMIL.MODEL.Image.MIL.CLAM.pl"
    }

    if model_name not in model_classes:
        raise ValueError(f"Model name '{model_name}' is not available")

    module_path = model_classes[model_name]
    module = __import__(module_path, fromlist=[''])
    model_class = getattr(module, f"pl_{model_name}")

    if model_name in ["ABMIL", "DSMIL"]:
        return model_class(
            data_paras=dataset_paras,
            opt_paras=optloss_paras,
            trainer_paras=train_paras,
            model_para=model_para
        )
    elif model_name == "AttentionMIL":
        return model_class(dataset_paras=dataset_paras, paras=model_para)
    else:
        return model_class(paras=model_para)

    

def _deprecated_create_img_model_(train_paras:PLTrainerParas,
                     dataset_paras:DatasetParas,
                     optloss_paras:OptLossParas,
                     model_para):
    """
    create img pl model
    """
    #--------> get model paras
    model_name = train_paras.model_name
    #--------> create model
    if model_name in aviliable_mil_models.keys():
        if model_name == "ABMIL":
            from HistoMIL.MODEL.Image.MIL.ABMIL.pl import pl_ABMIL
            pl_model = pl_ABMIL(data_paras=dataset_paras,
                            opt_paras=optloss_paras,
                            trainer_paras=train_paras,
                            model_para=model_para)
        elif model_name == "TransMIL":
            from HistoMIL.MODEL.Image.MIL.TransMIL.pl import pl_TransMIL
            # pl_model = pl_TransMIL(data_paras=dataset_paras,
            #                 opt_paras=optloss_paras,
            #                 trainer_paras=train_paras,
            #                 model_para=model_para)
            pl_model = pl_TransMIL(paras=model_para)
            
        elif model_name == "DSMIL":
            from HistoMIL.MODEL.Image.MIL.DSMIL.pl import pl_DSMIL
            pl_model = pl_DSMIL(data_paras=dataset_paras,
                            opt_paras=optloss_paras,
                            trainer_paras=train_paras,
                            model_para=model_para)
        elif model_name == 'Transformer':
            from HistoMIL.MODEL.Image.MIL.Transformer.pl import pl_Transformer
            # pdb.set_trace()
            pl_model = pl_Transformer(paras=model_para)
            # pdb.set_trace()
        elif model_name == 'AttentionMIL':
            from HistoMIL.MODEL.Image.MIL.AttentionMIL.pl import pl_AttentionMIL

            pl_model = pl_AttentionMIL(dataset_paras=dataset_paras, 
                                       paras=model_para)
            # pdb.set_trace()
        elif model_name == 'CAMIL':
            from HistoMIL.MODEL.Image.MIL.CAMIL.pl import pl_CAMIL
            pl_model = pl_CAMIL(paras=model_para)
        elif model_name == 'DTFD-MIL':
            from HistoMIL.MODEL.Image.MIL.DTFD_MIL.pl import pl_DTFDMIL
            pl_model = pl_DTFDMIL(paras=model_para)
        elif model_name == 'GraphTransformer':
            from HistoMIL.MODEL.Image.MIL.GraphTransformer.pl import pl_GraphTransformer
            pl_model = pl_GraphTransformer(paras=model_para)
        elif model_name == 'TransMILMultimodal':
            from HistoMIL.MODEL.Image.MIL.TransMILMultimodal.pl import pl_TransMILMultimodal
            pl_model = pl_TransMILMultimodal(paras=model_para)

    else:
        raise ValueError("model name not availiable")
    return pl_model

    
def create_img_mode_paras(train_paras:PLTrainerParas,):
    #--------> get model paras
    model_name = train_paras.model_name
    #--------> create model
    if model_name in aviliable_mil_models.keys():
        if model_name == "ABMIL":
            from HistoMIL.MODEL.Image.MIL.ABMIL.paras import AttMILParas
            model_paras = AttMILParas()
        elif model_name == "TransMIL":
            from HistoMIL.MODEL.Image.MIL.TransMIL.paras import TransMILParas
            model_paras = TransMILParas()
        elif model_name == "DSMIL":
            from HistoMIL.MODEL.Image.MIL.DSMIL.paras import DSMILParas
            model_paras = DSMILParas()
    else:
        raise ValueError("model name not aviliable")

    return model_paras