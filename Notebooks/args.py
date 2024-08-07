import argparse
# import datetime
import os
import random 
import numpy as np
import torch
import ast
import pdb

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

    # pdb.set_trace()
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
        if not isinstance(val, int):
            raise argparse.ArgumentTypeError(f"Value {val} for key {key} in the dictionary is not an integer")
    return value

def dict_type_mil(string):
    try:
        value = ast.literal_eval(string)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dict value: {string}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"Value {value} is not a dict")
    for key, val in value.items():
        # if not isinstance(key, int):
        #     raise argparse.ArgumentTypeError(f"Key {key} in the dictionary is not a int")
        if not isinstance(val, int):
            raise argparse.ArgumentTypeError(f"Value {val} for key {key} in the dictionary is not an integer")
    return value

def str_to_int_list(string):
    return [int(num) for num in string.split(',')]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args_split_array_job():
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
        "--cohort-dir", type=str, help='Experiment directory. ')
    parser.add_argument(
        "--task-additional-idx", type=str, nargs='+', default=None, help='additional column names of biomarkers of interest.')
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to ensure reproducibility")
    
    parser.add_argument(
        "--array-split-idx",type=int, help="Index indicating in how many parts will dataset be split for passing into array jobs in the cluster, e.g., 5 for splitting dataset into 5 parts")

    args = parser.parse_args()

    
    # assert args.data_dir.endswith(os.path.sep)
    assert args.cohort_dir.endswith(os.path.sep)

    # args.conf_version = args.data_dir.split(os.path.sep)[-2]
    # args.name = args.name + f"-{args.conf_version}"

    seed_everything(args.seed)

    return args


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
        "--transformations", type=str, default='only_naive_transforms_tensor', help="Tranformations for data augmentation, default is only_naive_transforms_tensor")
    

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to ensure reproducibility")
    # parser.add_argument(
    #     "--id", type=str, default='0001', help="Unique ID of the user")
    # parser.add_argument(
    #     "--username", default='draco',type=str, help="Unique name of the experimenter")
    parser.add_argument(
        "--api-dir",type=str, help="Directory where API.env for storing API keys is saved")
    parser.add_argument(
        "--array-job-idx",type=int, default=None, help="Index of the split dataset file. OPTIONAL, this is for array job submission in the cluster")
    parser.add_argument(
        "--k-fold", default=0, type=int, help='Number of folds for cross-validation, e.g. 3 for 3-fold cross-validation. Default is 0, no cross-validation')
    
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
        "--label-dict", type=dict_type_mil, default=None, help="Dictionary mapping target values to binary values")
    parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--concepts-name", nargs='+', default=['slide', 'patch', 'feature'], help="Name of concepts to be used, default is slide, patch, feature in this order")
    parser.add_argument(
        "--split-ratio", nargs='+', type=float, default=[0.99, 0.01], help="list of values indicating split ratio of the dataset (values MUST sum to 1), default, 99% training to 1% validation")
    ### Model arguments
    parser.add_argument(
        "--step-size", type=int, help="Step-size taken to crop the segmented tissue, and it's the SAME as patch size, e.g., 224")
    parser.add_argument(
        "--precomputed", default=None, type=str, help="Name of the backbone model used for feature extraction in the preprocessing stage. These precomputed features can be reused for training the MIL model, e.g. prov-gigapath")
    parser.add_argument(
        "--cohort-dir", type=str, help='Experiment directory. ')
    parser.add_argument(
        "--task-additional-idx", type=str, nargs='+', default=None, help='additional column names of biomarkers of interest.')
    parser.add_argument(
        "--num-workers", type=int, default=4, help='number of workers to specify in pytorch dataloader')
    parser.add_argument(
        "--mil-algorithm", type=str, help='Name of the MIL algorithm/model that is used, e.g., TransMIL, DSMIL, ABMIL, Transformer')
    parser.add_argument(
        "--pretrained-weights-dir", type=str, help='Directory where the pretrained-weights are stored for the MIL model')
    parser.add_argument(
        "--pretrained-weights-name", default=None, type=str, help='Filename (e.g. ending in .pth) of the pretrained weights to be loaded to the MIL model, e.g. MSI_high_CRC_model.pth')
    
    parser.add_argument(
        "--n-epochs", default=4, type=int, help='Maximum numer of epochs for training pytorch lightning model, default is 4. ')
    parser.add_argument(
        "--k-fold", default=0, type=int, help='Number of folds for cross-validation, e.g. 3 for 3-fold cross-validation. Default is 0, no cross-validation')
    
    parser.add_argument(
        "--monitor-metric", default='auroc_val', type=str, help='Performance metric to monitor by pytorch lightning which decides saved checkpoint, e.g. loss_val, auroc_val')
    
    parser.add_argument("--efficient-finetuning", type=str2bool, nargs='?', const=True, default=False, 
                    help="Set True to enable finetuning mlp_head and first input projection layer (default: True)")

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to ensure reproducibility")
    
    parser.add_argument(
        "--last-cv", default=0, type=int, help='Last cross-validation fold (count from zero) when experiment failed and from which to resume, e.g, . 3 if experiment failed at the 4th fold')
    parser.add_argument(
        "--ckpt-filename", default=None, type=str, help='Filename (exclude .ckpt extension) of the model checkpoint saved before the experiment crashed, e.g. attentionMIL_uni_32epoch_reruncv=3_epoch=23-auroc_val=0.65')
    
    
    # parser.add_argument(
    #     "--id", type=str, default='0001', help="Unique ID of the user")
    # parser.add_argument(
    #     "--username", default='draco',type=str, help="Unique name of the experimenter")
    # parser.add_argument(
    #     "--api-dir",type=str, help="Directory where API.env for storing API keys is saved")
    
    args = parser.parse_args()

    
    # assert args.data_dir.endswith(os.path.sep)
    assert args.cohort_dir.endswith(os.path.sep)
    # assert args.pretrained_weights_dir.endswith(os.path.sep)

    # args.conf_version = args.data_dir.split(os.path.sep)[-2]
    # args.name = args.name + f"-{args.conf_version}"

    seed_everything(args.seed)

    return args


def get_args_mil_rerun():
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
        "--label-dict", type=dict_type_mil, default=None, help="Dictionary mapping target values to binary values")
    parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--concepts-name", nargs='+', default=['slide', 'patch', 'feature'], help="Name of concepts to be used, default is slide, patch, feature in this order")
    parser.add_argument(
        "--split-ratio", nargs='+', type=float, default=[0.99, 0.01], help="list of values indicating split ratio of the dataset (values MUST sum to 1), default, 99% training to 1% validation")
    ### Model arguments
    parser.add_argument(
        "--step-size", type=int, help="Step-size taken to crop the segmented tissue, and it's the SAME as patch size, e.g., 224")
    parser.add_argument(
        "--precomputed", default=None, type=str, help="Name of the backbone model used for feature extraction in the preprocessing stage. These precomputed features can be reused for training the MIL model, e.g. prov-gigapath")
    parser.add_argument(
        "--cohort-dir", type=str, help='Experiment directory. ')
    parser.add_argument(
        "--task-additional-idx", type=str, nargs='+', default=None, help='additional column names of biomarkers of interest.')
    parser.add_argument(
        "--num-workers", type=int, default=4, help='number of workers to specify in pytorch dataloader')
    parser.add_argument(
        "--mil-algorithm", type=str, help='Name of the MIL algorithm/model that is used, e.g., TransMIL, DSMIL, ABMIL, Transformer')
    parser.add_argument(
        "--pretrained-weights-dir", type=str, help='Directory where the pretrained-weights are stored for the MIL model')
    parser.add_argument(
        "--pretrained-weights-name", default=None, type=str, help='Filename (e.g. ending in .pth) of the pretrained weights to be loaded to the MIL model, e.g. MSI_high_CRC_model.pth')
    
    parser.add_argument(
        "--n-epochs", default=4, type=int, help='Maximum numer of epochs for training pytorch lightning model, default is 4. ')
    parser.add_argument(
        "--k-fold", default=0, type=int, help='Number of folds for cross-validation, e.g. 3 for 3-fold cross-validation. Default is 0, no cross-validation')
    
    parser.add_argument(
        "--last-cv", default=0, type=int, help='Last cross-validation fold (count from zero) when experiment failed and from which to resume, e.g, . 3 if experiment failed at the 4th fold')
    parser.add_argument(
        "--ckpt-filename", default=None, type=str, help='Filename (exclude .ckpt extension) of the model checkpoint saved before the experiment crashed, e.g. attentionMIL_uni_32epoch_reruncv=3_epoch=23-auroc_val=0.65')
    
    parser.add_argument(
        "--monitor-metric", default='auroc_val', type=str, help='Performance metric to monitor by pytorch lightning which decides saved checkpoint, e.g. loss_val, auroc_val')
    
    parser.add_argument("--efficient-finetuning", type=str2bool, nargs='?', const=True, default=True, 
                    help="Set True to enable finetuning mlp_head and first input projection layer (default: True)")

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to ensure reproducibility")
    # parser.add_argument(
    #     "--id", type=str, default='0001', help="Unique ID of the user")
    # parser.add_argument(
    #     "--username", default='draco',type=str, help="Unique name of the experimenter")
    # parser.add_argument(
    #     "--api-dir",type=str, help="Directory where API.env for storing API keys is saved")
    
    args = parser.parse_args()

    
    # assert args.data_dir.endswith(os.path.sep)
    assert args.cohort_dir.endswith(os.path.sep)
    # assert args.pretrained_weights_dir.endswith(os.path.sep)

    # args.conf_version = args.data_dir.split(os.path.sep)[-2]
    # args.name = args.name + f"-{args.conf_version}"

    seed_everything(args.seed)

    return args

def get_args_mil_inference():
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
    parser.add_argument(
        "--n-epochs", default=None)
    parser.add_argument(
        "--pid-name", type=str, default='PatientID', help="Column name of patient ID, e.g. PatientID")
    parser.add_argument(
        "--targets-name", help="list of column names of the target prediction, e.g., g0_arrest hrd rs", nargs='+')
    parser.add_argument(
        "--label-dict", type=dict_type_mil, default=None, help="Dictionary mapping target values to binary values")
    parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--concepts-name", nargs='+', default=['slide', 'patch', 'feature'], help="Name of concepts to be used, default is slide, patch, feature in this order")
    parser.add_argument(
        "--split-ratio", nargs='+', type=float, default=[0.99, 0.01], help="list of values indicating split ratio of the dataset (values MUST sum to 1), default, 99% training to 1% validation")
    ### Model arguments
    parser.add_argument(
        "--step-size", type=int, help="Step-size taken to crop the segmented tissue, and it's the SAME as patch size, e.g., 224")
    parser.add_argument(
        "--precomputed", default=None, type=str, help="Name of the backbone model used for feature extraction in the preprocessing stage. These precomputed features can be reused for training the MIL model, e.g. prov-gigapath")
    parser.add_argument(
        "--cohort-dir", type=str, help='Experiment directory. ')
    parser.add_argument(
        "--task-additional-idx", type=str, nargs='+', default=None, help='additional column names of biomarkers of interest.')
    parser.add_argument(
        "--num-workers", type=int, default=4, help='number of workers to specify in pytorch dataloader')
    parser.add_argument(
        "--mil-algorithm", type=str, help='Name of the MIL algorithm/model that is used, e.g., TransMIL, DSMIL, ABMIL, Transformer')
    
    parser.add_argument(
        "--monitor-metric", default='auroc_val', type=str, help='Performance metric to monitor by pytorch lightning which decides saved checkpoint, e.g. loss_val, auroc_val')
    
    parser.add_argument("--efficient-finetuning", type=str2bool, nargs='?', const=True, default=False, 
                    help="Set True to enable finetuning mlp_head and first input projection layer (default: True)")

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to ensure reproducibility")
    
    parser.add_argument(
        "--ckpt-filenames", nargs='+', default=None, type=str, help='List of best model checkpoint filenames (exclude .ckpt extension) of model checkpoints saved after training finished, e.g. attentionMIL_uni_32epoch_reruncv=3_epoch=23-auroc_val=0.65')
    
    parser.add_argument(
        "--ckpt-filename", default=None)
    
    parser.add_argument(
        "--k-fold", default=5, type=int, help='Number of folds for cross-validation, e.g. 3 for 3-fold cross-validation. Default is 0, no cross-validation')
        
    parser.add_argument(
        "--pretrained-weights-dir", default=None)
    parser.add_argument(
        "--pretrained-weights-name", default=None)
    
    parser.add_argument(
        "--ensemble", action="store_true", help="Use this flag if you prefer ensemble predictions")
    
    args = parser.parse_args()

    
    # assert args.data_dir.endswith(os.path.sep)
    assert args.cohort_dir.endswith(os.path.sep)
    # assert args.pretrained_weights_dir.endswith(os.path.sep)

    # args.conf_version = args.data_dir.split(os.path.sep)[-2]
    # args.name = args.name + f"-{args.conf_version}"

    seed_everything(args.seed)

    return args