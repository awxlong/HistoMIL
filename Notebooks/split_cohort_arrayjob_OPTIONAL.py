"""
Preprocessing the WSIs in parallel
"""
### Setting path for HistoMIL
import os
import pdb
import sys
sys.path.append(os.getcwd())

import math
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' # avoid pandas warning
import torch
torch.multiprocessing.set_sharing_strategy('file_system') # avoid multiprocessing problem
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # stop skimage warning
def ignore_warnings(*args, **kwargs):
    pass
import pickle


# HistoMIL imports
from HistoMIL.EXP.paras.env import EnvParas
from HistoMIL.EXP.workspace.experiment import Experiment
from HistoMIL import logger
import logging
logger.setLevel(logging.INFO)

from args import get_args_split_array_job




def split_for_arrayjob(args):
    '''
    please see args.py for an explanation of what each argument means
    '''

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
    logger.info("initiating splitting ...")

    idx = args.array_split_idx
    cohort_name = preprocess_env.cohort_para.localcohort_name
    task_name = preprocess_env.cohort_para.task_name
    root = f'{args.cohort_dir}Data/'
    df1 = pd.read_csv(f'{root}{preprocess_env.cohort_para.cohort_file}')
    df2 = pd.read_csv(f'{root}{preprocess_env.cohort_para.task_file}')
    # pdb.set_trace()
    # Merge the dataframes on PatientID
    merged_df = pd.merge(df1, df2, on=preprocess_env.cohort_para.pid_name).drop_duplicates()

    # Calculate the number of rows for each part
    total_rows = len(merged_df)
    rows_per_part = math.ceil(total_rows / idx)

    # Split and save the data
    for i in range(idx):
        start_idx = i * rows_per_part
        end_idx = min((i + 1) * rows_per_part, total_rows)
        
        # Split the merged dataframe
        part_df = merged_df.iloc[start_idx:end_idx]
        
        # Split df1
        part_df1 = part_df[df1.columns]
        part_df1.to_csv(f'{root}local_cohort_{cohort_name}_{i+1}.csv', index=False)
        
        # Split df2
        part_df2 = part_df[['PatientID', f'{task_name}']]
        part_df2.to_csv(f'{root}{cohort_name}_{task_name}_{i+1}.csv', index=False)
        print(f"Files {root}{cohort_name}_{task_name}_{i+1}.csv and {root}local_cohort_{cohort_name}_{i+1}.csv have been split and saved successfully.")
        
    
def main():
    args = get_args_split_array_job()
    split_for_arrayjob(args)
if __name__ == "__main__":
    main()

