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
from HistoMIL.DATA.Database.data_aug import only_naive_transforms_tensor, no_transforms, only_naive_transforms
import logging
logger.setLevel(logging.INFO)

from args import get_args_preprocessing
from huggingface_hub import login
from dotenv import load_dotenv
from torchvision import transforms


import h5py
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
import numpy as np
from scipy.spatial import distance
import torch
import os
from sklearn import preprocessing

FEAT_DIMS = {
    'uni': 1024,
    'prov-gigapath': 1536,
    'ctranspath': 768,
    'resnet50': 2048
}

def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_available_device()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def compute_distances_in_chunks(X, chunk_size=1000):
    # chunks = pairwise_distances_chunked(X, metric='euclidean', n_jobs=1, working_memory=chunk_size)
    # pdb.set_trace()
    # distances = np.vstack(list(chunks)) # becomes boottleneck due to dtype 64
    n_samples = X.shape[0]
    distances = np.memmap('temp_distances.dat', dtype='float32', mode='w+', shape=(n_samples, n_samples))

    for i, chunk in enumerate(pairwise_distances_chunked(X, metric='euclidean', n_jobs=1, working_memory=chunk_size)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_samples)
        
        # Convert chunk to float32 to save memory
        chunk = chunk.astype(np.float32)
        pdb.set_trace()
        # Store the chunk in the memmap array
        distances[start:end, :] = chunk
    return distances

# def compute_distances_in_batches(X, batch_size=1000, dtype=np.float32):
#     # X = X.astype(dtype)  
#     n = X.shape[0]
#     distances = np.zeros((n, n), dtype=dtype)
#     for i in range(0, n, batch_size):
#         end = min(i + batch_size, n)
#         distances[i:end] = pairwise_distances(X[i:end], X, metric='euclidean')
#         yield distances # will distances have full rows such that argsort would be done in an entire row
#     # return distances

def compute_distances_in_batches(X, batch_size=1000, dtype=np.float32):
    # X = X.astype(dtype)  
    n = X.shape[0]
    # distances = np.zeros((n, n), dtype=dtype)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_distances = pairwise_distances(X[i:end], X, metric='euclidean')
        yield batch_distances


def compute_adj_coords(wsi_coords, wsi_feats, wsi_name, adj_coord_save_path, adj_matrix_save_path, force_recalc = False):
        # output_path_file = os.path.join(save_path + wsi_name + '.h5')
        # output_path_file = data_locs.abs_loc('feature') + f'{encoder}_adj_dictionary/{wsi_name}.h5'
        if not os.path.exists(f'{adj_matrix_save_path}{wsi_name}.pt') or force_recalc: 
             
            # patch_distances = pairwise_distances(wsi_coords, metric='euclidean', n_jobs=1) # array of shape (num_nodes, num_nodes)
            # pdb.set_trace()
            # wsi_coords = wsi_coords.astype(np.int32)
            # patch_distances = compute_distances_in_chunks(wsi_coords)
            # pdb.set_trace()
            # patch_distances = compute_distances_in_batches(wsi_coords)
            # pdb.set_trace()
            # neighbor_indices = np.argsort(patch_distances, axis=1)[:, :16]
            # n_rows, _ = patch_distances.shape
            # n_neighbors = 16
            # neighbor_indices = np.empty((n_rows, n_neighbors), dtype=np.int32)
            # for i in range(n_rows):
            #     sorted_indices = np.argsort(patch_distances[i])
            #     neighbor_indices[i] = sorted_indices[:n_neighbors]
            chunk_size = 4096
            n_neighbors = 16
            neighbor_indices = np.empty((wsi_coords.shape[0], n_neighbors), dtype=np.int32)
            for idx, batch_patch_distance in enumerate(compute_distances_in_batches(wsi_coords, batch_size=chunk_size)):
                # sorted_indices = np.argsort(batch_patch_distance)
                # neighbor_indices[idx] = sorted_indices[:n_neighbors]
                start_idx = idx * chunk_size
                end_idx = min((idx + 1) * chunk_size, wsi_coords.shape[0])
                
                for i, distances in enumerate(batch_patch_distance):
                    sorted_indices = np.argsort(distances)
                    neighbor_indices[start_idx + i] = sorted_indices[:n_neighbors]
            # pdb.set_trace()
            rows = np.asarray([[enum] * len(item) for enum, item in enumerate(neighbor_indices)]).ravel()
            columns = neighbor_indices.ravel()
            values = []
            coords = []
            for row, column in zip(rows, columns):
                    m1 = np.expand_dims(wsi_feats[int(row)], axis=0)
                    # pdb.set_trace()
                    m2 = np.expand_dims(wsi_feats[int(column)], axis=0)
                    value = distance.cdist(m1.reshape(1, -1), m2.reshape(1, -1), 'cosine')[0][0]
                    values.append(value)
                    coords.append((row, column))
            
            # mode = 'a'
            values = np.array(values, dtype=np.float32) # reduce memory
            values = np.reshape(values, (wsi_coords.shape[0], neighbor_indices.shape[1]))
            
            coords = np.array(coords)
            
            # asset_dict = {'adj_coords': coords, 'similarities': values, 'indices': neighbor_indices}

            # save_hdf5(adj_coord_save_path, asset_dict, attr_dict=None)

            ### compute adjacency matrix
            values = np.nan_to_num(values)

            Idx = neighbor_indices[:, :8]
            rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()

            columns = Idx.ravel()

            neighbor_matrix = values[:, 1:]

            normalized_matrix = preprocessing.normalize(neighbor_matrix, norm="l2")

            similarities = np.exp(-normalized_matrix)

            values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)

            values = values[:, :8]

            values = values.ravel().tolist()

            sparse_coords= list(zip(rows, columns))

            # sparse_matrix = torch.sparse_coo_tensor(sparse_coords, values, (wsi_feats2.shape[0], wsi_feats2.shape[0]))

            indices = torch.tensor(sparse_coords, dtype=torch.long).t()
            # values = torch.tensor(values, dtype=torch.float32)
            values = torch.FloatTensor(values)
            sparse_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([wsi_feats.shape[0], wsi_feats.shape[0]]))

            torch.save(sparse_matrix, f'{adj_matrix_save_path}{wsi_name}.pt')
            logger.info(f'Adjacency matrix stored at {adj_matrix_save_path}')
        else:
             logger.info(f'Adjacency matrix already exists at: {adj_matrix_save_path}{wsi_name}.pt')
        # return np.array(coords), values, neighbor_indices, sparse_matrix



def preprocess_adj_matrices(args):
   

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

    preprocess_env.cohort_para.update_localcohort = True
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
    logger.info("setup adjacency matrix computation")
    
    # 
    # pdb.set_trace()
    
    local_cohort_idx_file = pd.read_csv(f'{args.cohort_dir}Data/{preprocess_env.cohort_para.cohort_file}')
    wsi_coord_root = data_locs.abs_loc('patch') + f'{args.step_size}_{args.step_size}/'
    wsi_feats_root = data_locs.abs_loc('feature') + f'{args.backbone_name}/'

    h5_path_root = data_locs.abs_loc('feature') + f'{args.backbone_name}_adj_dictionary/'
    sparse_matrix_root = data_locs.abs_loc('feature') + f'{args.backbone_name}_adj_matrix/'

    os.makedirs(h5_path_root, exist_ok=True)
    os.makedirs(sparse_matrix_root, exist_ok=True)

    for i in range(local_cohort_idx_file.shape[0]):
        wsi_name = f'{local_cohort_idx_file.loc[i, "folder"]}.{local_cohort_idx_file.loc[i, "filename"]}'
        wsi_coords_name = f'{wsi_name}.h5'
        wsi_feats = f'{wsi_name}.pt'

        wsi_coords_dir = f'{wsi_coord_root}{wsi_coords_name}'
        wsi_feats_dir = f'{wsi_feats_root}{wsi_feats}'
        # pdb.set_trace()
        if os.path.exists(wsi_coords_dir) and os.path.exists(wsi_feats_dir):
                # Load the slide and its coordinates
                wsi_coordinates = h5py.File(wsi_coords_dir)
                wsi_coordinates = wsi_coordinates['coords']
                wsi_features = torch.load(wsi_feats_dir)  
                
                adj_coords_save_path = f'{h5_path_root}{wsi_name}.h5' # data_locs.abs_loc('feature') + f'{encoder}_adj_dictionary/{wsi_name}.h5'
                adj_matrix_save_path = f'{sparse_matrix_root}{wsi_name}.pt'
                
                # Process the slide and its coordinates
                compute_adj_coords(wsi_coords = wsi_coordinates, 
                                wsi_feats = wsi_features,
                                wsi_name = wsi_name,
                                adj_coord_save_path=adj_coords_save_path,
                                adj_matrix_save_path=adj_matrix_save_path,
                                force_recalc=False
                                )
            
            
        else:
            logger.info(f'{wsi_coords_dir} \n or {wsi_feats_dir} doesn not exist')
                # Do nothing and continue to the next iteration
            continue
    
def main():
    args = get_args_preprocessing()
    preprocess_adj_matrices(args)
if __name__ == "__main__":
    main()
