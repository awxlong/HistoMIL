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



# HistoMIL imports
from HistoMIL.EXP.paras.env import EnvParas
from HistoMIL.EXP.workspace.experiment import Experiment
from HistoMIL import logger
from HistoMIL.DATA.Database.data_aug import only_naive_transforms_tensor, no_transforms, only_naive_transforms
import logging
logger.setLevel(logging.INFO)

from args import get_args_preprocessing
from huggingface_hub import login
from dotenv import load_dotenv
from torchvision import transforms


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import os

import pandas as pd
from skimpy import skim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pdb
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
np.random.seed(42)
# import pandas_profiling

import pdb


# from pandas_profiling import ProfileReport


def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_available_device()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def replace_icd_o_3_site(df, column_name):
    icd_o_3_site_map = {
        'C18.0': 'Cecum',
        'C18.2': 'Ascending colon',
        'C18.3': 'Hepatic flexure of colon',
        'C18.4': 'Transverse colon',
        'C18.5': 'Splenic flexure of colon',
        'C18.6': 'Descending colon',
        'C18.7': 'Sigmoid colon',
        'C18.9': 'Colon, NOS (Not Otherwise Specified)',
        'C19.9': 'Rectosigmoid junction',
        'C20.9': 'Rectum, NOS',
        'C49.4': 'Connective and soft tissue of abdomen',
        'C80.9': 'Unknown primary site'
    }
    
    # Replace the codes with their meanings
    df[column_name] = df[column_name].map(icd_o_3_site_map).fillna(df[column_name])
    
    # Replace any remaining codes with 'Unknown code'
    # df[column_name] = df[column_name].apply(lambda x: 'Unknown code' if x.startswith('C') else x)
    
    return df


def replace_icd_o_3_histology(df, column_name):
    icd_o_3_histology_map = {
        '8140/3': 'Adenocarcinoma, NOS',
        '8480/3': 'Mucinous adenocarcinoma',
        '8260/3': 'Papillary adenocarcinoma, NOS',
        '8560/3': 'Adenosquamous carcinoma',
        '8574/3': 'Adenocarcinoma with neuroendocrine differentiation',
        '8255/3': 'Adenocarcinoma with mixed subtypes',
        '8010/3': 'Carcinoma, NOS',
        '8263/3': 'Adenocarcinoma in tubulovillous adenoma',
        '8211/3': 'Tubular adenocarcinoma'
    }
    
    # Replace the codes with their meanings
    df[column_name] = df[column_name].map(icd_o_3_histology_map).fillna(df[column_name])
    
    # Replace any remaining codes with 'Unknown histology'
    # df[column_name] = df[column_name].apply(lambda x: 'Unknown histology' if isinstance(x, str) and '/' in x else x)
    
    return df

def replace_unknown(df, strings_to_replace = ['?', 'Unknown/Invalid', 'Unknown']):

    return df.replace({col: {string: pd.NA for string in strings_to_replace} for col in df.columns}, inplace=False)

def imputation(split, train_split):
   mode_values = train_split.mode().iloc[0]

   imputed_split = split.fillna(mode_values)

   return imputed_split


def group_balance(split, train_df):
    """
    Replacement map is computed on training split
    Training-split-based replacement map is applied to desired split, i.e., train, test, val. 
    """
    ### based on training stats
    
    ### RACE IMBALANCE: White
    split['Race'] = split['Race'].replace({
        value: 'NON-WHITE' for value in split['Race'].unique() if value not in train_df['Race'].value_counts(normalize=True).head(1).index
        }
    )

    ### ICD O 3 HISTOLOGY IMBALANCE: Adenocarcinoma, NOS
    split['Icd O 3 Histology'] = split['Icd O 3 Histology'].replace({
        value: 'Other' for value in split['Icd O 3 Histology'].unique() if value not in train_df['Icd O 3 Histology'].value_counts(normalize=True).head(1).index
        }
    )

    ### ICD O 3 SITE IMBALANCE: beyond Rectum, NOS
    split['Icd O 3 Site'] = split['Icd O 3 Site'].replace({
        value: 'Other' for value in split['Icd O 3 Site'].unique() if value not in train_df['Icd O 3 Site'].value_counts(normalize=True).head(6).index
        }
    )

    ### RESIDUAL TUMOR IMBALANCE: R0
    split['Residual Tumor'] = split['Residual Tumor'].replace({
        value: 'Other Stage' for value in split['Residual Tumor'].unique() if value not in train_df['Residual Tumor'].value_counts(normalize=True).head(1).index
        }
    )    
    return split

def cast_category(split, train_cat_vars):
    split[train_cat_vars] = split[train_cat_vars].astype('category')
    # split = pd.get_dummies(split, columns=train_cat_vars, prefix= train_cat_vars)
    return split

def one_hot_encode(split, train_df):

    cats = [col for col in train_df.columns if train_df[col].dtypes == "category"]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first') # pad categories which values coincidentally don't appear in val/test sets
    encoder.fit(train_df[cats])
    split_encoded = encoder.transform(split[cats])
    encoded_cols = encoder.get_feature_names_out(cats)
    split_encoded_df = pd.DataFrame(split_encoded, columns=encoded_cols)

    # pdb.set_trace()
    split_final = pd.concat([split.reset_index(drop=True), split_encoded_df], axis=1).drop(cats, axis=1)
    
    return split_final

def preprocess(split, val_split, test_split):

    # global encoding icd_o_3 codes
    split = replace_icd_o_3_site(split, 'Icd O 3 Site')
    split = replace_icd_o_3_histology(split, 'Icd O 3 Histology')
    val_split = replace_icd_o_3_site(val_split, 'Icd O 3 Site')
    val_split = replace_icd_o_3_histology(val_split, 'Icd O 3 Histology')
    test_split = replace_icd_o_3_site(test_split, 'Icd O 3 Site')
    test_split = replace_icd_o_3_histology(test_split, 'Icd O 3 Histology')

    # global preprocessing: replace 'Unknown' with NA
    split = replace_unknown(split)
    val_split = replace_unknown(val_split)
    test_split = replace_unknown(test_split)
    
    # global preprocessing: drop columns with >45% NA rate 
    columns_to_drop = ['Weight', 'Circumferential Resection Margin', 'Perineural Invasion Present', 'Microsatellite Instability', 'Colon Polyps Present', 'Radiation Therapy', 'Primary Therapy Outcome Success', 'Non Nodal Tumor Deposits', 'Kras Mutation Found', 'Braf Gene Analysis Result', 'Postoperative Rx Tx', 'New Tumor Event After Initial Treatment', 'Prescribed Dose', 'Number Cycles', 'Measure Of Response']
    split = split.drop(columns=columns_to_drop)
    val_split = val_split.drop(columns=columns_to_drop)
    test_split = test_split.drop(columns=columns_to_drop)

    # global: clip biologically implausible Cea levels
    cea_column = 'Preoperative Pretreatment Cea Level'  # Replace with your actual column name

    # Determine clipping thresholds
    lower_bound = split[cea_column].quantile(0.01)  # 1st percentile
    upper_bound = split[cea_column].quantile(0.95)  # 95th percentile

    # Clip values outside the normal range
    split['Preoperative Pretreatment Cea Level'] = split['Preoperative Pretreatment Cea Level'].clip(lower=lower_bound, upper=upper_bound)
    val_split['Preoperative Pretreatment Cea Level'] = val_split['Preoperative Pretreatment Cea Level'].clip(lower=lower_bound, upper=upper_bound)
    test_split['Preoperative Pretreatment Cea Level'] = test_split['Preoperative Pretreatment Cea Level'].clip(lower=lower_bound, upper=upper_bound)
    
    # global
    split['Other Dx'] = split['Other Dx'].replace({
        'Yes, History of Synchronous/Bilateral Malignancy': 'Yes'
    })
    val_split['Other Dx'] = val_split['Other Dx'].replace({
        'Yes, History of Synchronous/Bilateral Malignancy': 'Yes'
    })
    test_split['Other Dx'] = test_split['Other Dx'].replace({
        'Yes, History of Synchronous/Bilateral Malignancy': 'Yes'
    })
    
    # train-set dependent preprocessing: use train-set's column-wise mode to impute missing values
    split = imputation(split, train_split = split)
    val_split = imputation(val_split, train_split = split)
    test_split = imputation(test_split, train_split = split)


    # train-set dependent preprocessing: clip categories based on train-set's column-wise, categorical variables' relative proportion
    split = group_balance(split, train_df = split)
    val_split = group_balance(val_split, train_df = split)
    test_split = group_balance(test_split, train_df = split)
    
    # global: ignore 'g0-arrest'
    split = split.loc[:, split.columns != 'g0_arrest']
    val_split = val_split.loc[:, val_split.columns != 'g0_arrest']
    test_split = test_split.loc[:, test_split.columns != 'g0_arrest']


    # global preprocessing: convert the object data type into categorical data type AFTER one-hot encoding which convert data types
    categorical_vars = [col for col in split.columns if split[col].dtypes == 'object' and col != 'PatientID'] # + ['g0_arrest'] 
    split = cast_category(split, train_cat_vars=categorical_vars)
    val_split = cast_category(val_split, train_cat_vars=categorical_vars)
    test_split = cast_category(test_split, train_cat_vars=categorical_vars)
    # pdb.set_trace()

    # global: one-hot encode
    X_train_split = one_hot_encode(split, split)
    X_val_split = one_hot_encode(val_split, split)
    X_test_split = one_hot_encode(test_split, split)
    # pdb.set_trace()
    
    
    # train-set dependent normalization
    scaler = MinMaxScaler()
    float_columns = ['Age At Initial Pathologic Diagnosis', 'Lymph Node Examined Count', 'Preoperative Pretreatment Cea Level'] # X_train_split.select_dtypes(include=['float64']).columns
    X_train_split[float_columns] = scaler.fit_transform(X_train_split[float_columns])
    X_val_split[float_columns] = scaler.transform(X_val_split[float_columns])
    X_test_split[float_columns] = scaler.transform(X_test_split[float_columns])


    # X_train_split = split.loc[:, split.columns != 'g0_arrest']
    # X_val_split = val_split.loc[:, val_split.columns != 'g0_arrest']
    # X_test_split = test_split.loc[:, test_split.columns != 'g0_arrest']
    # y_split = split['readmitted']

    features_selected = ['PatientID', 'Age At Initial Pathologic Diagnosis',
    'Lymph Node Examined Count',
    'Preoperative Pretreatment Cea Level',
    'Gender_MALE',
    'Race_WHITE',
    'Other Dx_Yes',
    'Pathologic Stage_Stage II',
    'Pathologic Stage_Stage IIA',
    'Pathologic Stage_Stage IIB',
    'Pathologic Stage_Stage III',
    'Pathologic Stage_Stage IIIB',
    'Pathologic Stage_Stage IIIC',
    'Pathologic Stage_Stage IV',
    'Pathologic Stage_Stage IVA',
    'Icd O 3 Histology_Other',
    'Icd O 3 Site_Cecum',
    'Icd O 3 Site_Colon, NOS (Not Otherwise Specified)',
    'Icd O 3 Site_Other',
    'Icd O 3 Site_Rectosigmoid junction',
    'Icd O 3 Site_Rectum, NOS',
    'Icd O 3 Site_Sigmoid colon',
    'Person Neoplasm Cancer Status_WITH TUMOR',
    'Venous Invasion_YES',
    'Lymphatic Invasion_YES',
    'History Of Colon Polyps_YES',
    'Residual Tumor_R0',
    'Loss Expression Of Mismatch Repair Proteins By Ihc_YES']
    
    X_train_split = X_train_split[features_selected]
    X_val_split = X_val_split[features_selected]
    # pdb.set_trace()
    X_test_split = X_test_split[features_selected]

    return X_train_split, X_val_split, X_test_split # , y_split


def preprocess_clinical(args):
   

    preprocess_env = EnvParas()


    preprocess_env.exp_name = args.exp_name         # e.g. "debug_preprocess"
    preprocess_env.project = args.project_name      # e.g. "test-project" 
    preprocess_env.entity = args.wandb_entity_name  # make sure it's initialized to an existing wandb entity
    #----------------> cohort
    preprocess_env.cohort_para.localcohort_name = args.localcohort_name # "BRCA"
    preprocess_env.cohort_para.task_name = args.task_name               # e.g "DNAD"
    if args.array_job_idx:
        preprocess_env.cohort_para.cohort_file = f'local_cohort_{preprocess_env.cohort_para.localcohort_name}_{args.array_job_idx}.csv'                         # e.g. local_cohort_CRC.csv, this is created automatically, and contains folder, filename, slide_nb, tissue_nb, etc. 
        preprocess_env.cohort_para.task_file = f'{preprocess_env.cohort_para.localcohort_name}_{preprocess_env.cohort_para.task_name}_{args.array_job_idx}.csv' # e.g. CRC_g0_arrest.csv, which has PatientID matched with g0_arrest labels
    else:
        preprocess_env.cohort_para.cohort_file = f'local_cohort_{preprocess_env.cohort_para.localcohort_name}.csv'                         # e.g. local_cohort_CRC.csv, this is created automatically, and contains folder, filename, slide_nb, tissue_nb, etc. 
        preprocess_env.cohort_para.task_file = f'{preprocess_env.cohort_para.localcohort_name}_{preprocess_env.cohort_para.task_name}.csv' # e.g. CRC_g0_arrest.csv, which has PatientID matched with g0_arrest labels
    preprocess_env.cohort_para.pid_name = args.pid_name     # "PatientID" # the column with which to merge tables
    preprocess_env.cohort_para.targets = args.targets_name  # e.g. "g0_arrest"  # the column name of interest, supply as a list
    preprocess_env.cohort_para.targets_idx = 0              # don't know what this is
    preprocess_env.cohort_para.label_dict = args.label_dict # e.g. "{'HRD':0,'HRP':1}" # SINGLE quotations for the keys
    preprocess_env.cohort_para.task_additional_idx = args.task_additional_idx # ["g0_score"] # if CRC_g0_arrest.csv has other biomarkers of interest, name them in this variable, default None. 

    preprocess_env.cohort_para.update_localcohort = False
    #----------------> pre-processing

    # #----------------> model
    # slide-level parameters
    print(preprocess_env.collector_para.slide)

    # tissue-level parameters
    print(preprocess_env.collector_para.tissue)

    # patch-level parameters
    preprocess_env.collector_para.patch.step_size = args.step_size # e.g. 224 # ASSUME this also decides the size of patch, although you can change this
    preprocess_env.collector_para.patch.patch_size = (args.step_size, args.step_size) 
    preprocess_env.collector_para.patch.from_contours = True
    print(preprocess_env.collector_para.patch)

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
    logger.info("setup clinical feature computation and storage")
    
    exp = Experiment(env_paras=preprocess_env)
    exp.setup_machine(machine=machine,user=user)
    logger.info("setup data")
    exp.paras.trainer_para.k_fold = args.k_fold 
    exp.init_cohort()
    
    label_idx = exp.paras.cohort_para.targets[exp.paras.cohort_para.targets_idx]
    exp.data_cohort.show_taskcohort_stat(label_idx=label_idx)
    exp.split_train_test()  # updated to split into train, valid, test
    # pdb.set_trace()
    # /Users/awxlong/Desktop/my-studies/hpc_exps/Data/CRC_WSI_clinical_g0_features.csv

    # Create a directory to store the tensors if it doesn't exist
    output_dir_root = f'{preprocess_env.data_locs.abs_loc("feature")}clinical/'
    

    for kfold in range(args.k_fold):
        df = pd.read_csv(f'{preprocess_env.exp_locs.root}Data/{args.localcohort_name}_WSI_clinical_{args.task_name}_features.csv')
                  
        exp.get_i_th_fold(kfold)
        
        train_patient_ids, val_patient_ids, test_patient_ids = exp.data_cohort.data['train']['PatientID'], \
                                                               exp.data_cohort.data['valid']['PatientID'], \
                                                               exp.data_cohort.data['test']['PatientID']
        df_train, df_val, df_test = df[df['PatientID'].isin(train_patient_ids)], \
                                    df[df['PatientID'].isin(val_patient_ids)], \
                                    df[df['PatientID'].isin(test_patient_ids)]
        # pdb.set_trace()
        X_train, X_val, X_test = preprocess(df_train, df_val, df_test)
        # pdb.set_trace()
        # List of DataFrames and their corresponding names
        dataframes = [('train', X_train), ('validation', X_val), ('test', X_test)]

        for split_name, df in dataframes:
            print(f"Processing {split_name} split...")

            if split_name == 'train' or split_name == 'validation':

                cv_dir = f'{output_dir_root}cv{kfold}/'
                os.makedirs(cv_dir, exist_ok=True)
                for _, row in df.iterrows():
                    # Get the patient ID from the first column
                    patient_id = row.iloc[0]
                    
                    # Extract the features (remaining 32 columns)
                    features = row.iloc[1:].values
                    
                    # Convert features to numpy array of float type
                    features_array = np.array(features, dtype=np.float32)
                    
                    # Convert the numpy array to a PyTorch tensor
                    tensor = torch.from_numpy(features_array)

                    # Create the filename using the patient ID
                    filename = f"{patient_id}.pt"
                    file_path = os.path.join(cv_dir, filename)
                    
                    # Check if the file already exists
                    if os.path.exists(file_path):
                        # Load the existing tensor
                        existing_tensor = torch.load(file_path)
                        
                        # Check if the existing tensor is the same as the new tensor
                        if torch.equal(existing_tensor, tensor):
                            print(f"Tensor for patient ID {patient_id} already exists and is identical. Skipping...")
                            continue
                    
                    # Save the tensor to a .pt file
                    torch.save(tensor, file_path)
                    print(f"Saved tensor for patient ID {patient_id} to {file_path}")
            elif split_name == 'test':

                test_dir = f'{output_dir_root}test/'
                os.makedirs(test_dir, exist_ok=True)
                for _, row in df.iterrows():
                    # Get the patient ID from the first column
                    patient_id = row.iloc[0]
                    
                    # Extract the features (remaining 32 columns)
                    features = row.iloc[1:].values
                    
                    # Convert features to numpy array of float type
                    features_array = np.array(features, dtype=np.float32)
                    
                    # Convert the numpy array to a PyTorch tensor
                    tensor = torch.from_numpy(features_array)
                    # Create the filename using the patient ID
                    filename = f"{patient_id}.pt"
                    file_path = os.path.join(test_dir, filename)
                    # pdb.set_trace()
                    # Check if the file already exists
                    if os.path.exists(file_path):
                        # Load the existing tensor
                        existing_tensor = torch.load(file_path)
                        
                        # Check if the existing tensor is the same as the new tensor
                        if torch.equal(existing_tensor, tensor):
                            print(f"Tensor for patient ID {patient_id} already exists and is identical. Skipping...")
                            continue
                    
                    # Save the tensor to a .pt file
                    torch.save(tensor, file_path)
                    print(f"Saved tensor for patient ID {patient_id} to {file_path}")

                
    # pdb.set_trace()
def main():
    args = get_args_preprocessing()
    preprocess_clinical(args)
if __name__ == "__main__":
    main()
