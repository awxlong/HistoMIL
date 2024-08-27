"""
Main file to run a MIL pipeline which consists of training, cross-validation, inference and interpretability through heatmaps or integrated gradients
"""

#-----IMPORTS
#--------------------------> base env setting
# avoid pandas warning
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
# avoid multiprocessing problem
import torch
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')
#--------------------------> logging setup
import logging
logging.basicConfig(
    level=logging.INFO,
    format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d|%H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)
import os
import pdb
import sys
sys.path.append(os.getcwd())

import pickle
from dotenv import load_dotenv
import wandb

from args import get_args_mil

### Getting parameters for MIL model architectures/algorithms
from HistoMIL.MODEL.Image.MIL.TransMIL.paras import TransMILParas
from HistoMIL.MODEL.Image.MIL.TransMILMultimodal.paras import TransMILMultimodalParas
from HistoMIL.MODEL.Image.MIL.TransMILRegression.paras import TransMILRegressionParas
from HistoMIL.MODEL.Image.MIL.DSMIL.paras import DSMILParas
from HistoMIL.MODEL.Image.MIL.Transformer.paras import  DEFAULT_TRANSFORMER_PARAS
from HistoMIL.MODEL.Image.MIL.TransformerMultimodal.paras import  TransformerMultimodalParas
from HistoMIL.MODEL.Image.MIL.TransformerRegression.paras import  TransformerRegressionParas
from HistoMIL.MODEL.Image.MIL.AttentionMIL.paras import  DEFAULT_Attention_MIL_PARAS
from HistoMIL.MODEL.Image.MIL.CAMIL.paras import  CAMILParas, custom_camil_collate
from HistoMIL.MODEL.Image.MIL.DTFD_MIL.paras import  DTFD_MILParas
from HistoMIL.MODEL.Image.MIL.GraphTransformer.paras import  GraphTransformerParas
from HistoMIL.MODEL.Image.MIL.DTFDTransMIL.paras import  DTFDTransMILParas
from HistoMIL.MODEL.Image.MIL.CLAM.paras import CLAMParas
### Getting environment parameters
from HistoMIL.EXP.paras.env import EnvParas
from HistoMIL.EXP.workspace.experiment import Experiment
from HistoMIL.EXP.paras.trainer import get_pl_trainer_additional_paras
#------IMPORTS

### Feature encoders
MDL_TO_FEATURE_DIMS = {
    'prov-gigapath': 1536, 
    'resnet18': 512,
    'uni': 1024,
    'resnet50': 2048,
}

def run_mil(args):

    #-----Initialization of MIL models

    # TransMIL with Nystrom attention: https://github.com/szc19990412/TransMIL
    model_para_transmil = TransMILParas()
    model_para_transmil.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    model_para_transmil.n_classes=1
    model_para_transmil.lr_scheduler_config = {'T_max':args.n_epochs, 
                                                'eta_min':1e-6}
    model_para_transmil.epoch = args.n_epochs

    # TransMILMultimodal with clinical features
    DEFAULT_MULTIMODAL_TRANSMIL_PARAS = TransMILMultimodalParas()
    DEFAULT_MULTIMODAL_TRANSMIL_PARAS.epoch = args.n_epochs
    DEFAULT_MULTIMODAL_TRANSMIL_PARAS.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]

    # TransMILRegression
    DEFAULT_TRANSMIL_REGRESSION = TransMILRegressionParas()
    DEFAULT_TRANSMIL_REGRESSION.epoch = args.n_epochs
    ### DEFAULT_TRANSMIL_REGRESSION.theshold = args.threshold
    DEFAULT_TRANSMIL_REGRESSION.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]

    # DSMIL: https://github.com/binli123/dsmil-wsi
    model_para_dsmil = DSMILParas()
    model_para_dsmil.feature_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    model_para_dsmil.p_class = 2
    model_para_dsmil.b_class = 2
    model_para_dsmil.dropout_r = 0.5

    # Transformer with self-attention: https://github.com/peng-lab/HistoBistro
    DEFAULT_TRANSFORMER_PARAS.input_dim =  MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_TRANSFORMER_PARAS.pretrained_weights_dir = args.pretrained_weights_dir
    DEFAULT_TRANSFORMER_PARAS.pretrained_weights = args.pretrained_weights_name    # default is MSI_high_CRC_model.pth 
    DEFAULT_TRANSFORMER_PARAS.selective_finetuning = args.efficient_finetuning
    DEFAULT_TRANSFORMER_PARAS.epoch = args.n_epochs
    DEFAULT_TRANSFORMER_PARAS.lr_scheduler_config = {'T_max':args.n_epochs, 
                                                    'eta_min':1e-6}
    
    # TransformerMultimodal fusion of clinical features
    DEFAULT_MULTIMODAL_TRANSFORMER_PARAS = TransformerMultimodalParas()
    DEFAULT_MULTIMODAL_TRANSFORMER_PARAS.input_dim =  MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_MULTIMODAL_TRANSFORMER_PARAS.epoch = args.n_epochs

    # TransformerRegression 
    DEFAULT_TRANSFORMER_REGRESSION_PARAS = TransformerRegressionParas()
    DEFAULT_TRANSFORMER_REGRESSION_PARAS.input_dim =  MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_TRANSFORMER_REGRESSION_PARAS.epoch = args.n_epochs
    
    # AttentionMIL: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    DEFAULT_Attention_MIL_PARAS.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_Attention_MIL_PARAS.epoch = args.n_epochs

    # CAMIL: https://github.com/olgarithmics/ICLR_CAMIL
    DEFAULT_CAMIL_PARAS = CAMILParas()
    DEFAULT_CAMIL_PARAS.input_shape = MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_CAMIL_PARAS.epoch = args.n_epochs

    # DTFD_MIL: https://github.com/hrzhang1123/DTFD-MIL
    DEFAULT_DTFD_MIL_PARAS = DTFD_MILParas()
    DEFAULT_DTFD_MIL_PARAS.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_DTFD_MIL_PARAS.epoch = args.n_epochs
    
    # GraphTransformer: https://github.com/vkola-lab/tmi2022
    DEFAULT_GRAPHTRANSFORMER_PARAS = GraphTransformerParas()
    DEFAULT_GRAPHTRANSFORMER_PARAS.n_features = MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_GRAPHTRANSFORMER_PARAS.epoch = args.n_epochs

    # CLAM: https://github.com/mahmoodlab/CLAM
    DEFAULT_CLAM_PARAS = CLAMParas()
    DEFAULT_CLAM_PARAS.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_CLAM_PARAS.epoch = args.n_epochs

    # Hybrid DTFD-MIL-TransMIL
    DEFAULT_DTFDMIL_TRANSMIL_PARAS = DTFDTransMILParas()
    DEFAULT_DTFDMIL_TRANSMIL_PARAS.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_DTFDMIL_TRANSMIL_PARAS.epoch = args.n_epochs

    
    model_name = args.mil_algorithm  # options are "TransMIL", "TransMILRegression", "TransMILMultimodal", "ABMIL", "DSMIL", "Transformer", "TransformerRegression", "TransformerMultimodal", 'AttentionMIL', "CAMIL", "DTFD_MIL", "GraphTransformer", "CLAM", "DTFD-MIL-TransMIL"

    model_para_settings = {"TransMIL":model_para_transmil,
                           'TransMILRegression': DEFAULT_TRANSMIL_REGRESSION,
                           "DSMIL":model_para_dsmil,
                           'Transformer':DEFAULT_TRANSFORMER_PARAS,
                           'TransformerMultimodal': DEFAULT_MULTIMODAL_TRANSFORMER_PARAS,
                           'TransformerRegression': DEFAULT_TRANSFORMER_REGRESSION_PARAS,
                           'AttentionMIL': DEFAULT_Attention_MIL_PARAS,
                           'CAMIL': DEFAULT_CAMIL_PARAS,
                           'DTFD_MIL': DEFAULT_DTFD_MIL_PARAS,
                           'GraphTransformer': DEFAULT_GRAPHTRANSFORMER_PARAS,
                           'TransMILMultimodal': DEFAULT_MULTIMODAL_TRANSMIL_PARAS,
                           'DTFDTransMIL': DEFAULT_DTFDMIL_TRANSMIL_PARAS,
                           'CLAM': DEFAULT_CLAM_PARAS
                           } 

    #----------------> environment parameters
    mil_env = EnvParas()
    mil_env.exp_name = f'{args.exp_name}'    # name of the experiment which is also the checkpoint name for the model that's going to be saved in SavedModel/ e.g. "debug_process"
    mil_env.project = args.project_name      # e.g. "test-project" 
    mil_env.entity = args.wandb_entity_name  # make sure it's initialized to an existing WandB entity
    #----------------> cohort
    mil_env.cohort_para.localcohort_name = args.localcohort_name # "BRCA"
    mil_env.cohort_para.task_name = args.task_name               # e.g "DNAD"
    mil_env.cohort_para.cohort_file = f'local_cohort_{mil_env.cohort_para.localcohort_name}.csv' 
    mil_env.cohort_para.task_file = mil_env.cohort_para.task_file = f'{mil_env.cohort_para.localcohort_name}_{mil_env.cohort_para.task_name}.csv' # e.g. CRC_g0_arrest.csv, which has PatientID matched with g0_arrest labels   
    mil_env.cohort_para.pid_name = args.pid_name                 # the column with which to merge tables, by default it's matched by "PatientID"
    mil_env.cohort_para.targets = args.targets_name              # e.g. "g0_arrest"  # the column name of interest, which can be supplied as a list of column names
    mil_env.cohort_para.targets_idx = 0
    mil_env.cohort_para.label_dict = args.label_dict             # e.g. "{'HRD':0,'HRP':1}" # SINGLE quotations for the keys
    mil_env.cohort_para.task_additional_idx = args.task_additional_idx # ["g0_score"] # if CRC_g0_arrest.csv has other biomarkers of interest, name them in this variable, default None. 
    mil_env.cohort_para.in_domain_split_seed = 42                # for consistent in-domain train-val-test split and REPRODUCIBILITY

    #---------------> collector parameters and trainer / analyzer
    if args.precomputed:
        # extract precomputed features after running pre_processing
        mil_env.trainer_para.use_pre_calculated = True                              # for loading precomputed patch features from pre_processing
        mil_env.trainer_para.backbone_name = args.precomputed
        mil_env.collector_para.feature.model_name = args.precomputed                # name of the feature encoder, e.g. prov-gigapath if we want to reuse precomputed prov-gigapath features 
        mil_env.collector_para.feature.img_size = (args.step_size,  args.step_size) 
        mil_env.collector_para.patch.step_size = args.step_size
        mil_env.collector_para.patch.patch_size = (args.step_size,  args.step_size) # patch size of tesselated WSI. we assume no overlap and square patches 
    else:
        # extract patch features on the go with a standard resnet18
        mil_env.trainer_para.backbone_name = "resnet18"
        mil_env.trainer_para.additional_pl_paras.update({"accumulate_grad_batches":8})
        mil_env.trainer_para.label_format = "int"
    
    
    mil_env.cohort_para.update_localcohort = False # don't update local_cohort file during training, cv, inference nor interpretability analysis
    
    #----------------> dataset parameters
    mil_env.dataset_para.dataset_name = args.dataset_name       # e.g. "DNAD_L2"
    mil_env.dataset_para.concepts = args.concepts_name          # default is computing ["slide","tissue","patch","feature"]
    mil_env.dataset_para.split_ratio = args.split_ratio         # train-test split ratio, which must sum to one, and by default it's [0.9,0.1]
    mil_env.dataset_para.num_workers = args.num_workers         # num_workers for Pytorch dataloader, e.g. 8
    ### additional feature extraction
    if args.mil_algorithm == 'CAMIL' or args.mil_algorithm == 'GraphTransformer':
        # Spatial constraints
        mil_env.dataset_para.additional_feature = 'AdjMatrix'
        mil_env.dataset_para.add_dataloader = {'collate_fn':custom_camil_collate}
    if 'Multimodal' in args.mil_algorithm:
        # Loading clinical features
        mil_env.dataset_para.additional_feature = 'Clinical'
    if 'Regression' in args.mil_algorithm:
        # Loading biomarker scores 
        mil_env.dataset_para.additional_feature = 'Regression'

    #----------------> model parameters
    mil_env.trainer_para.model_name = model_name
    mil_env.trainer_para.model_para = model_para_settings[model_name]
    
    # --------------> Logging metrics
    mil_env.trainer_para.ckpt_format = "_{epoch:02d}-{auroc_val:.4f}" # additional substring that's appended to self.exp_name to be the filename of .ckpt file stored in SavedModel/
    
    #  --------------> pytorch lightning parameters
    mil_env.trainer_para.ckpt_para = { #-----------> paras for pytorch_lightning.callbacks.ModelCheckpoint
                    "save_top_k":1,
                    "mode":"max" if args.monitor_metric == 'auroc_val' or args.monitor_metric == 'f1_val' else 'min',
                    "monitor":args.monitor_metric,} # decides how checkpointing tracks model weights, e.g. whether maximizing AUROC or F1 
    
    mil_env.opt_para.max_epochs = args.n_epochs 

    #-----------------> init machine and person
    machine_cohort_loc = f"{args.cohort_dir}/User/{args.localcohort_name}_machine_config.pkl"
    with open(machine_cohort_loc, "rb") as f:   # Unpickling
        [data_locs,exp_locs,machine,user] = pickle.load(f)
    mil_env.data_locs = data_locs
    mil_env.exp_locs = exp_locs
    
    # --------------> pytorch lightning updated parameters
    if args.ckpt_filename:
        ### Resume from checkpoing for continuing crashed experiments
        mdl_ckpt_root = mil_env.exp_locs.abs_loc('saved_models')
        mil_env.trainer_para.additional_pl_paras=get_pl_trainer_additional_paras(args.mil_algorithm)
        mil_env.trainer_para.additional_pl_paras.update(
                      {'resume_from_checkpoint':  f'{mdl_ckpt_root}{args.ckpt_filename}.ckpt'}
                      )
    else:
        ### Start experiment from scratch
        mil_env.trainer_para.additional_pl_paras= get_pl_trainer_additional_paras(args.mil_algorithm)

    # --------------> experiment initialization
    logging.info("setup MIL experiment")
    exp = Experiment(env_paras=mil_env)
    exp.setup_machine(machine=machine,user=user)
    logging.info("setup data")
    exp.init_cohort()
    logging.info("setup trainer..")
    ### cross-validation parameters
    exp.paras.trainer_para.k_fold = args.k_fold  # number of CV folds, default is 5
    print(exp.paras.trainer_para)
    # pdb.set_trace()

    if args.train:
        if exp.paras.trainer_para.k_fold > 1:
            exp.setup_cv_experiment(last_cv=args.last_cv,
                                    main_data_source="slide",
                                    need_train=True)
        else:
            exp.setup_experiment(main_data_source="slide",
                                need_train=True)
            exp.exp_worker.train()
            val_results = exp.exp_worker.validate()
            print(val_results)

    elif args.inference:
        exp.ensemble_test(ckpt_filenames=args.ckpt_filenames)
    
    elif args.interpretability:
        if 'Multimodal' in args.mil_algorithm:
            exp.ensemble_integrated_gradients(ckpt_filenames=args.ckpt_filenames, ensemble=args.ensemble)

        exp.ensemble_heatmap(ckpt_filenames=args.ckpt_filenames, ensemble=args.ensemble)



    

if __name__ == '__main__':
    args = get_args_mil()
    run_mil(args)
    

    


    
    

            


