"""
example of run mil experiment
"""
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
from HistoMIL.MODEL.Image.MIL.DSMIL.paras import DSMILParas
from HistoMIL.MODEL.Image.MIL.Transformer.paras import  DEFAULT_TRANSFORMER_PARAS
from HistoMIL.MODEL.Image.MIL.AttentionMIL.paras import  DEFAULT_Attention_MIL_PARAS

from HistoMIL.EXP.paras.env import EnvParas
from HistoMIL.EXP.workspace.experiment import Experiment


# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy


MDL_TO_FEATURE_DIMS = {
    'prov-gigapath': 1536, 
    'resnet18': 512,
    'uni': 1024,
    'resnet50': 2048,
}

from datetime import datetime

def run_mil(args):
    #--------------------------> task setting
    #--------------------------> model setting

    # for transmil
    model_para_transmil = TransMILParas()
    model_para_transmil.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    model_para_transmil.n_classes=1
    model_para_transmil.lr_scheduler_config = {'T_max':args.n_epochs, 
                                                'eta_min':1e-6}
    model_para_transmil.epoch = args.n_epochs

    # for dsmil
    model_para_dsmil = DSMILParas()
    model_para_dsmil.feature_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    model_para_dsmil.p_class = 2
    model_para_dsmil.b_class = 2
    model_para_dsmil.dropout_r = 0.5

    # for Transformer
    DEFAULT_TRANSFORMER_PARAS.input_dim =  MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_TRANSFORMER_PARAS.pretrained_weights_dir = args.pretrained_weights_dir
    DEFAULT_TRANSFORMER_PARAS.pretrained_weights = args.pretrained_weights_name    # default is MSI_high_CRC_model.pth 
    DEFAULT_TRANSFORMER_PARAS.selective_finetuning = args.efficient_finetuning
    DEFAULT_TRANSFORMER_PARAS.epoch = args.n_epochs
    DEFAULT_TRANSFORMER_PARAS.lr_scheduler_config = {'T_max':args.n_epochs, 
                                                    'eta_min':1e-6}
    # pdb.set_trace()
    DEFAULT_Attention_MIL_PARAS.input_dim = MDL_TO_FEATURE_DIMS[args.precomputed]
    DEFAULT_Attention_MIL_PARAS.epoch = args.n_epochs

    
    model_name = args.mil_algorithm  # options are "TransMIL", "ABMIL", "DSMIL" or "Transformer", 'AttentionMIL'

    model_para_settings = {"TransMIL":model_para_transmil,
                           "DSMIL":model_para_dsmil,
                           'Transformer':DEFAULT_TRANSFORMER_PARAS,
                           'AttentionMIL': DEFAULT_Attention_MIL_PARAS,
                           'CAMIL': None,
                           'DTFD-MIL': None} 

    #--------------------------> parameters
    
    gene2k_env = EnvParas()
    gene2k_env.exp_name = f'{args.exp_name}' # f'{args.exp_name}_{datetime.now().strftime("%m%d_%H%M")}'         # name of the experiment which is also the checkpoint model that's going to be saved in SavedModel/ e.g. "debug_process + time.time()"
    gene2k_env.project = args.project_name      # e.g. "test-project" 
    gene2k_env.entity = args.wandb_entity_name  # make sure it's initialized to an existing wandb entity
    #----------------> cohort
    gene2k_env.cohort_para.localcohort_name = args.localcohort_name # "BRCA"
    gene2k_env.cohort_para.task_name = args.task_name               # e.g "DNAD"
    gene2k_env.cohort_para.cohort_file = f'local_cohort_{gene2k_env.cohort_para.localcohort_name}.csv' 
    gene2k_env.cohort_para.task_file = gene2k_env.cohort_para.task_file = f'{gene2k_env.cohort_para.localcohort_name}_{gene2k_env.cohort_para.task_name}.csv' # e.g. CRC_g0_arrest.csv, which has PatientID matched with g0_arrest labels   
    gene2k_env.cohort_para.pid_name = args.pid_name     # "PatientID" # the column with which to merge tables
    gene2k_env.cohort_para.targets = args.targets_name  # e.g. "g0_arrest"  # the column name of interest, supply as a list
    gene2k_env.cohort_para.targets_idx = 0
    gene2k_env.cohort_para.label_dict = args.label_dict # e.g. "{'HRD':0,'HRP':1}" # SINGLE quotations for the keys
    gene2k_env.cohort_para.task_additional_idx = args.task_additional_idx # ["g0_score"] # if CRC_g0_arrest.csv has other biomarkers of interest, name them in this variable, default None. 
    gene2k_env.cohort_para.in_domain_split_seed = 42                      # for consisten in-domain train-val-test split


    #---------------> collector parameters and trainer / analyzer
    if args.precomputed:
        gene2k_env.trainer_para.use_pre_calculated = True ### FOR LOADING COMPUTED FEATURES
        gene2k_env.trainer_para.backbone_name = args.precomputed
        gene2k_env.collector_para.feature.model_name = args.precomputed # if i want to reuse precomputed prov-gigapath # this model_name is different to the model_name below
        gene2k_env.collector_para.patch.step_size = args.step_size
        gene2k_env.collector_para.patch.patch_size = (args.step_size,  args.step_size)
    else:
        gene2k_env.trainer_para.backbone_name = "resnet18"
        gene2k_env.trainer_para.additional_pl_paras.update({"accumulate_grad_batches":8})
        gene2k_env.trainer_para.label_format = "int"#"one_hot"  # change here for regression?
    gene2k_env.cohort_para.update_localcohort = True ## update local_cohort file
    #----------------> pre-processing
    #----------------> dataset
    gene2k_env.dataset_para.dataset_name = args.dataset_name # e.g. "DNAD_L2"
    gene2k_env.dataset_para.concepts = args.concepts_name    # default ["slide","patch","feature"]
    gene2k_env.dataset_para.split_ratio = args.split_ratio   # default [0.8,0.2]
    gene2k_env.dataset_para.num_workers = args.num_workers   # num_workers for dataloader, e.g. 8
    #----------------> model
    gene2k_env.trainer_para.model_name = model_name
    gene2k_env.trainer_para.model_para = model_para_settings[model_name]
    
    # --------------> Logging metrics
    gene2k_env.trainer_para.ckpt_format = "_{epoch:02d}-{auroc_val:.4f}" # additional substring that's appended to self.exp_name to be the filename of .ckpt file stored in SavedModel/

    gene2k_env.trainer_para.ckpt_para = { #-----------> paras for pytorch_lightning.callbacks.ModelCheckpoint
                    "save_top_k":1,
                    "mode":"max" if args.monitor_metric == 'auroc_val' else 'min',
                    "monitor":args.monitor_metric,}
    
    gene2k_env.trainer_para.additional_pl_paras={
                #---------> paras for pytorch lightning trainner
                "accumulate_grad_batches":8, # mil need accumulated grad
                "accelerator":"auto",        #accelerator='gpu', devices=1,
                'precision': 16,             # Use mixed precision
                'enable_progress_bar': True, 
                'enable_model_summary': True,
                # 'gradient_clip_val': 0.5,
                # 'plugins': [DDPStrategy(find_unused_parameters=False)],
                # 'strategy':'ddp',
            }

    gene2k_env.opt_para.max_epochs = args.n_epochs # lower than https://www.cell.com/cancer-cell/fulltext/S1535-6108(23)00278-7

    # pdb.set_trace()

    #k_fold = None
    

    #--------------------------> init machine and person
    machine_cohort_loc = f"{args.cohort_dir}/User/{args.localcohort_name}_machine_config.pkl"
    with open(machine_cohort_loc, "rb") as f:   # Unpickling
        [data_locs,exp_locs,machine,user] = pickle.load(f)
    gene2k_env.data_locs = data_locs
    gene2k_env.exp_locs = exp_locs
    #---------------> wandb 
    # wandb.setup(settings=wandb.Settings(
    #     _disable_stats=True,
    #     # disable_git=True,
    #     api_key=user.wandb_api_key  
    # ))

    # wandb.init(project=gene2k_env.project, 
    #            entity=gene2k_env.entity)   ### ADD THIS IN CLUSTER
    
    #--------------------------> setup experiment
    logging.info("setup MIL experiment")
    
    exp = Experiment(env_paras=gene2k_env)
    exp.setup_machine(machine=machine,user=user)
    logging.info("setup data")
    exp.init_cohort()
    logging.info("setup trainer..")
    
    exp.paras.trainer_para.k_fold = args.k_fold 
    print(exp.paras.trainer_para)

    # pdb.set_trace()
    if exp.paras.trainer_para.k_fold > 1:
       exp.setup_cv_experiment(main_data_source="slide",
                            need_train=True)
    else:
        
        
        exp.setup_experiment(main_data_source="slide",
                            need_train=True)
        # exp.paras.trainer_para.backbone_name = args.precomputed
        # pdb.set_trace()

        exp.exp_worker.train()

        val_results = exp.exp_worker.validate()

        print(val_results)

    

if __name__ == '__main__':
    args = get_args_mil()
    run_mil(args)
    

    


    
    

            


