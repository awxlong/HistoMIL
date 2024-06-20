import argparse
# import datetime
import os
import random 
import numpy as np
import torch
import ast


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

def dict_type(string):
    try:
        value = ast.literal_eval(string)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dict value: {string}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"Value {value} is not a dict")
    for key, val in value.items():
        if not isinstance(key, str):
            raise argparse.ArgumentTypeError(f"Key {key} in the dictionary is not a string")
        if not isinstance(val, int):
            raise argparse.ArgumentTypeError(f"Value {val} for key {key} in the dictionary is not an integer")
    return value

def str_to_int_list(string):
    return [int(num) for num in string.split(',')]

def get_args_preprocessing():
    parser = argparse.ArgumentParser()
    ### logging arguments
    parser.add_argument(
        "--exp-name", type=str, help="Name of the experiment, e.g., preprocessing")
    parser.add_argument(
        "--project-name", type=str, help="Title of the project, e.g., persistence")
    parser.add_argument(
        "--wandb-entity-name", type=str, help="Name of an EXISTING wandb entity, e.g., cell-x")
    parser.add_argument(
        "--localcohort-name", type=str, help="Name of the cohort, e.g., BRCA for breast cancer")
    parser.add_argument(
        "--task-name", type=str, help="Name describing task, e.g., g0-arrest")
    ### Dataset arguments
    # parser.add_argument(
    #     "--cohort-file-dir", type=str, help="Directory to file describing the slides' directories and associated labels")
    parser.add_argument(
        "--pid-name", type=str, default='PatientID', help="Column name of patient ID, e.g. PatientID")
    parser.add_argument(
        "--targets-name", help="list of column names of the target prediction, e.g., g0_arrest hrd rs", nargs='+')
    parser.add_argument(
        "--label-dict", type=dict_type, help="Dictionary mapping target values to binary values, e.g. {'HRD':0,'HRP':1}, SINGLE quotations for keys")
    parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--concepts-name", nargs='+', default=['slide', 'tissue', 'patch', 'feature'], help="Name of the preprocessing stages, default is slide, tissue, patch, feature in this order")
    parser.add_argument(
        "--split-ratio", nargs='+', type=float, default=[0.99, 0.01], help="list of values indicating split ratio of the dataset (values MUST sum to 1), default, 99% training to 1% validation")
    ### Model arguments
    parser.add_argument(
        "--step-size", type=int, help="Step-size taken to crop the segmented tissue, and it's the SAME as patch size, e.g., 224")
    parser.add_argument(
        "--backbone-name", type=str, help="Name of the backbone model for feature extraction from cropped patches, e.g., prov-gigapath")
    parser.add_argument(
        "--cohort-dir", type=str, help='Experiment directory. ')
    parser.add_argument(
        "--task-additional-idx", type=str, nargs='+', default=None, help='additional column names of biomarkers of interest.')
    
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to ensure reproducibility")
    # parser.add_argument(
    #     "--id", type=str, default='0001', help="Unique ID of the user")
    # parser.add_argument(
    #     "--username", default='draco',type=str, help="Unique name of the experimenter")
    # parser.add_argument(
    #     "--api-dir", default='/Users/awxlong/Desktop/my-studies/hpc_exps/HistoMIL/',type=str, help="Directory where API.env for storing API keys is saved")
    
    args = parser.parse_args()

    
    # assert args.data_dir.endswith(os.path.sep)
    assert args.cohort_dir.endswith(os.path.sep)

    # args.conf_version = args.data_dir.split(os.path.sep)[-2]
    # args.name = args.name + f"-{args.conf_version}"

    seed_everything(args.seed)

    return args


def get_args_mil():
    parser = argparse.ArgumentParser()
    ### logging arguments
    parser.add_argument(
        "--exp-name", type=str, help="Name of the experiment, e.g., preprocessing")
    parser.add_argument(
        "--project-name", type=str, help="Title of the project, e.g., persistence")
    parser.add_argument(
        "--wandb-entity-name", type=str, help="Name of an EXISTING wandb entity, e.g., cell-x")
    parser.add_argument(
        "--localcohort-name", type=str, help="Name of the cohort, e.g., BRCA for breast cancer")
    parser.add_argument(
        "--task-name", type=str, help="Name describing task, e.g., g0-arrest")
    ### Dataset arguments
    # parser.add_argument(
    #     "--cohort-file-dir", type=str, help="Directory to file describing the slides' directories and associated labels")
    parser.add_argument(
        "--pid-name", type=str, default='PatientID', help="Column name of patient ID, e.g. PatientID")
    parser.add_argument(
        "--targets-name", help="list of column names of the target prediction, e.g., g0_arrest hrd rs", nargs='+')
    parser.add_argument(
        "--label-dict", type=dict_type, help="Dictionary mapping target values to binary values, e.g. {'HRD':0,'HRP':1}, SINGLE quotations for keys")
    parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--concepts-name", nargs='+', default=['slide', 'tissue', 'patch', 'feature'], help="Name of the preprocessing stages, default is slide, tissue, patch, feature in this order")
    parser.add_argument(
        "--split-ratio", nargs='+', type=float, default=[0.99, 0.01], help="list of values indicating split ratio of the dataset (values MUST sum to 1), default, 99% training to 1% validation")
    ### Model arguments
    parser.add_argument(
        "--step-size", type=int, help="Step-size taken to crop the segmented tissue, and it's the SAME as patch size, e.g., 224")
    parser.add_argument(
        "--backbone-name", type=str, help="Name of the backbone model for feature extraction from cropped patches, e.g., prov-gigapath")
    parser.add_argument(
        "--cohort-dir", type=str, help='Experiment directory. ')
    parser.add_argument(
        "--task-additional-idx", type=str, nargs='+', default=None, help='additional column names of biomarkers of interest.')
    
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to ensure reproducibility")
    # parser.add_argument(
    #     "--id", type=str, default='0001', help="Unique ID of the user")
    # parser.add_argument(
    #     "--username", default='draco',type=str, help="Unique name of the experimenter")
    # parser.add_argument(
    #     "--api-dir", default='/Users/awxlong/Desktop/my-studies/hpc_exps/HistoMIL/',type=str, help="Directory where API.env for storing API keys is saved")
    
    args = parser.parse_args()

    
    # assert args.data_dir.endswith(os.path.sep)
    assert args.cohort_dir.endswith(os.path.sep)

    # args.conf_version = args.data_dir.split(os.path.sep)[-2]
    # args.name = args.name + f"-{args.conf_version}"

    seed_everything(args.seed)

    return args