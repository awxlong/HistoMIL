import argparse
# import datetime
import os
import random 
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args_machine_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort-name", type=str, help="Name of the cohort, e.g., BRCA for breast cancer")
    parser.add_argument(
        "--data-dir", type=str, help="Directory to WSI data where subdirectories will be created")
    parser.add_argument(
        "--exp-dir", type=str, help="Directory to experiment-related files. Directory MUST be the parent directory which has the HistoMIL package")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to ensure reproducibility")
    parser.add_argument(
        "--id", type=str, default='0001', help="Unique ID of the user")
    parser.add_argument(
        "--username", default='draco',type=str, help="Unique name of the experimenter")
    parser.add_argument(
        "--api-dir", default='/Users/awxlong/Desktop/my-studies/hpc_exps/HistoMIL/',type=str, help="Directory where API.env for storing API keys is saved")
    
    args = parser.parse_args()

    
    assert args.data_dir.endswith(os.path.sep)
    assert args.exp_dir.endswith(os.path.sep)

    # args.conf_version = args.data_dir.split(os.path.sep)[-2]
    # args.name = args.name + f"-{args.conf_version}"

    seed_everything(args.seed)

    return args