"""
Preprocessing the WSIs, which include tissue segmentation, patching (also called tiling or tessellation) and feature extraction
"""
### Setting path for HistoMIL
import os
import pdb
import sys
sys.path.append(os.getcwd())

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' # avoid pandas warning
import torch
torch.multiprocessing.set_sharing_strategy('file_system') # avoid multiprocessing problem
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # stop skimage warning
import imageio.core.util
import skimage 
def ignore_warnings(*args, **kwargs):
    pass
imageio.core.util._precision_warn = ignore_warnings
import pickle
import timm


# HistoMIL imports
from HistoMIL.EXP.paras.env import EnvParas
from HistoMIL.EXP.workspace.experiment import Experiment
from HistoMIL import logger
import logging
logger.setLevel(logging.INFO)

from args import get_args_preprocessing
from huggingface_hub import login
from dotenv import load_dotenv




def preprocessing(args):

    load_dotenv(dotenv_path=f'{args.api_dir}API.env')
    hf_api_key = os.getenv('HF_READ_KEY')
    login(token=hf_api_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKBONES = {
    'uni': timm.create_model("hf_hub:MahmoodLab/UNI", pretrained=True).to(device),
    'prov-gigapath' : timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
    }
    

    preprocess_env = EnvParas()


    preprocess_env.exp_name = args.exp_name         # e.g. "debug_preprocess"
    preprocess_env.project = args.project_name      # e.g. "test-project" 
    preprocess_env.entity = args.wandb_entity_name  # make sure it's initialized to an existing wandb entity
    #----------------> cohort
    preprocess_env.cohort_para.localcohort_name = args.localcohort_name # "BRCA"
    preprocess_env.cohort_para.task_name = args.task_name               # e.g "DNAD"
    preprocess_env.cohort_para.cohort_file = f'local_cohort_{preprocess_env.cohort_para.localcohort_name}.csv'                         # e.g. local_cohort_CRC.csv, this is created automatically, and contains folder, filename, slide_nb, tissue_nb, etc. 
    preprocess_env.cohort_para.task_file = f'{preprocess_env.cohort_para.localcohort_name}_{preprocess_env.cohort_para.task_name}.csv' # e.g. CRC_g0_arrest.csv, which has PatientID matched with g0_arrest labels
    preprocess_env.cohort_para.pid_name = args.pid_name     # "PatientID" # the column with which to merge tables
    preprocess_env.cohort_para.targets = args.targets_name  # e.g. "g0_arrest"  # the column name of interest, supply as a list
    preprocess_env.cohort_para.targets_idx = 0              # don't know what this is
    preprocess_env.cohort_para.label_dict = args.label_dict # e.g. "{'HRD':0,'HRP':1}" # SINGLE quotations for the keys
    preprocess_env.cohort_para.task_additional_idx = args.task_additional_idx # ["g0_score"] # if CRC_g0_arrest.csv has other biomarkers of interest, name them in this variable, default None. 

    #preprocess_env.cohort_para.update_localcohort = True
    #----------------> pre-processing

    # #----------------> model
    # slide-level parameters
    print(preprocess_env.collector_para.slide)

    # tissue-level parameters
    print(preprocess_env.collector_para.tissue)

    # patch-level parameters
    preprocess_env.collector_para.patch.step_size = args.step_size # e.g. 224 # ASSUME this also decides the size of patch, although you can change this
    preprocess_env.collector_para.patch.patch_size = (args.step_size, args.step_size) 
    print(preprocess_env.collector_para.patch)

    # feature-extraction parameters
    # by default uses resnet18
    if args.backbone_name:
        preprocess_env.collector_para.feature.model_name = args.backbone_name                # e.g. 'prov-gigapath'
        preprocess_env.collector_para.feature.model_instance = BACKBONES[args.backbone_name] # timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    print(preprocess_env.collector_para.feature)

    #----------------> dataset
    preprocess_env.dataset_para.dataset_name = args.dataset_name # e.g. "DNAD_L2"
    preprocess_env.dataset_para.concepts = args.concepts_name    # default ['slide', 'tissue', 'patch', 'feature']
    preprocess_env.dataset_para.split_ratio = args.split_ratio   # e.g [0.99,0.01]
    
    
    machine_cohort_loc = f"{args.cohort_dir}/User/{args.localcohort_name}_machine_config.pkl"
    with open(machine_cohort_loc, "rb") as f:   # Unpickling
        [data_locs, exp_locs, machine,user] = pickle.load(f)
    preprocess_env.data_locs = data_locs
    preprocess_env.exp_locs = exp_locs
    
    #--------------------------> setup experiment
    logger.info("setup preprocessing experiment")
    
    exp = Experiment(env_paras=preprocess_env)
    exp.setup_machine(machine=machine,user=user)
    logger.info("setup data")
    exp.init_cohort()
    logger.info("pre-processing..")
    # pdb.set_trace()
    exp.cohort_slide_preprocessing(concepts = preprocess_env.dataset_para.concepts,
                                    is_fast = True, force_calc = False)

def main():
    args = get_args_preprocessing()
    preprocessing(args)
if __name__ == "__main__":
    main()


## SAMPLE COMMAND FOR CALLING THIS FUNCTION
# ## specify gigapath
# python HistoMIL/Notebooks/pre_processing.py --exp-name 'preprocessing-debug' --project-name 'g0-arrest' --wandb-entity-name 'cell-x' --localcohort-name 'CRC' --task-name 'g0-arrest' --pid-name 'PatientID' --targets-name 'g0_arrest' --cohort-dir '/Users/awxlong/Desktop/my-studies/hpc_exps/' --split-ratio 0.99 0.01 --step-size 224 --backbone-name 'prov-gigapath'



# #################----> for ssl
    # preprocess_env.trainer_para.method_type = "patch_learning"
    # preprocess_env.trainer_para.model_name = "moco" # 
    # from HistoMIL.MODEL.Image.SSL.paras import SSLParas
    # preprocess_env.trainer_para.model_para = SSLParas()
    # preprocess_env.trainer_para.model_para.ssl_dataset_para.batch_size = 16
    # preprocess_env.trainer_para.model_para.ssl_dataset_para.label_dict = {"HRD":0,"HRP":1}
    # preprocess_env.trainer_para.model_para.ssl_dataset_para.example_file = "example/example.png"
    # preprocess_env.trainer_para.model_para.ssl_dataset_para.is_weight_sampler = True
    # preprocess_env.trainer_para.model_para.ssl_dataset_para.force_balance_val = True
    # preprocess_env.trainer_para.model_para.ssl_dataset_para.add_dataloader = {
    #                                                     "pin_memory":True,
    #                                                     "drop_last":True,
    #                                                     }

    # from HistoMIL.DATA.Database.data_aug import SSL_DataAug
    # # specifu data aug or use default can be found at paras
    # preprocess_env.trainer_para.model_para.ssl_dataset_para.img_size = (512,512)
    # add_data_aug_paras = preprocess_env.trainer_para.model_para.ssl_dataset_para.add_data_aug_paras
    # trans_factory = SSL_DataAug(**add_data_aug_paras)
    # preprocess_env.trainer_para.model_para.ssl_dataset_para.transfer_fn = trans_factory.get_trans_fn
    # #----------------> trainer or analyzer
    # preprocess_env.trainer_para.label_format = "int"#"one_hot" 
    # preprocess_env.trainer_para.additional_pl_paras={
    #                 #---------> paras for pytorch lightning trainner
    #                 "accumulate_grad_batches":16, # mil need accumulated grad
    #                 "accelerator":"auto",#accelerator='gpu', devices=1,
    #             }
    # #preprocess_env.trainer_para.with_logger = None #without wandb to debug
    #--------------------------> init machine and person
    #--------------------------> init machine and person