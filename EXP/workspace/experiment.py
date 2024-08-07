"""
To initialise env paras and save to some folder
"""
from HistoMIL import logger
from HistoMIL.EXP.workspace.env import Person,Machine
from HistoMIL.DATA.Cohort.data import DataCohort
from HistoMIL.DATA.Cohort.utils import cohort_pre_processing
from HistoMIL.DATA.Slide.collector.data_collector import read_wsi_collector

from HistoMIL.EXP.paras.env import EnvParas
from HistoMIL.EXP.trainer.slide import pl_slide_trainer
from HistoMIL.EXP.paras.trainer import get_pl_trainer_additional_paras
from HistoMIL.DATA.Slide.concepts.WholeSlideImage import WholeSlideImageHeatmap, to_percentiles, top_k, screen_coords, sample_indices, sample_rois

from sklearn.metrics import f1_score, roc_auc_score, r2_score

import wandb
import numpy as np
import pandas as pd
import torch
import h5py
import pickle
import os
import pdb


    
class Experiment:
    def __init__(self,env_paras:EnvParas) -> None:
        self.paras = env_paras  
        self.cohort_para = env_paras.cohort_para

        self.exp_name = self.paras.exp_name
        self.project = self.paras.project
        self.entity   = self.paras.entity
        logger.info(f"Exp:: Start Environment {self.exp_name}")

    ################################################################
    #   Machine related
    ################################################################
    def setup_machine(self,machine:Machine,user:Person):
        """
        Setup machine related parameters
        in:
            data_locs:Locations: file structure for datas
            exp_locs:Locations:  file structure for exp
            usr:Person: user info for current exp
            device:str: target device
        add:
            self.machine:Machine: machine obj for current env
            self.data_locs
            self.exp_locs
        """
        logger.info("Exp:: Set up machine")
        self.machine = machine
        self.user = user

    ################################################################
    #   Cohort related to dataloader
    ################################################################
    """
    *.svs...localfile=>LocalCohort=>csv
    csv,index file=>TaskCohort=>datalist=>dataset=>dataloader
    """

    #------> pre processing function
    def cohort_slide_preprocessing(self,
                        df=None, # None means locol cohort,
                        concepts:list=["slide","tissue","patch",
                                        "feature"],
                        is_fast:bool=True,force_calc:bool=False,):
        """
        Preprocess cohort data 
        in:
            df:pd.DataFrame: data to preprocess if None, use all data in task cohort
            concepts:list: concepts to preprocess
            is_fast:bool: if True, segmentation will use smallest size in svs file 
            force_calc:bool: if True, will force to recalculate
        """
        self.backbone_net = self.paras.trainer_para.backbone_name#["backbone"]
        if df is not None: df = df 
        else:
            logger.info(f"Exp:: pre-processing all data in local cohort for concepts {concepts}")
            df = self.data_cohort.local_cohort.table.df
        # pdb.set_trace()
        logger.info(f"Exp:: pre-processing with paras:\n{self.paras.collector_para}")
        cohort_pre_processing(  df=df,
                                data_locs=self.machine.data_locs,#model only consider one data loc
                                collector_paras=self.paras.collector_para,
                                concepts=concepts,
                                fast_process=is_fast,
                                force_calc=force_calc,
                                )
    #------> other function for data
    def init_cohort(self):
        """
        initialise data cohort: including local cohort and task cohort
        """
        if self.paras.main_data_source =="slide":
            logger.info("Exp:: Initialise slide-based data cohort")
            # self.paras.collector_para.feature.model_name = self.paras.trainer_para.backbone_name
            self.data_cohort = DataCohort(data_locs=self.machine.data_locs,
                                        exp_locs=self.machine.exp_locs,
                                        cohort_para=self.paras.cohort_para,)
            self.task_concepts = self.paras.dataset_para.concepts#["task_concepts"]
            
            self.data_cohort.setup_localcohort()
            if self.paras.cohort_para.cohort_file is not None:
                self.data_cohort.setup_taskcohort( 
                                    task_concepts=self.task_concepts)
            
        elif self.paras.main_data_source == "omic":
            logger.info("Exp:: Initialise omic cohort.")
            self.data_cohort = DataCohort(data_locs=self.machine.data_locs,
                                        exp_locs=self.machine.exp_locs,
                                        cohort_para=self.paras.cohort_para,)
            
        else:   
            raise ValueError(f"main data source {self.paras.main_data_source} not supported.(slide or omic)")
        # initial some paras for dataset from cohort paras
        if self.paras.cohort_para.label_dict:
            # CLASSIFICATION
            self.paras.dataset_para.label_dict = self.paras.cohort_para.label_dict
            self.paras.cohort_para.category_nb = len(self.paras.cohort_para.label_dict.keys()) 
            self.paras.dataset_para.class_nb = self.paras.cohort_para.category_nb 
            

    def split_train_test(self,df=None):
        """
        split train and test data
        df:pd.DataFrame: data to split if None, use all data in task cohort
        """
        ratio = self.paras.dataset_para.split_ratio
        label_name = self.data_cohort.task_cohort.labels_name[0]
        # shuffle original data frome
        self.data_cohort.cohort_shuffle()
        # pdb.set_trace()
        self.data_cohort.split_train_phase(ratio=ratio,
                                            label_name=label_name,
                                            K_fold=self.paras.trainer_para.k_fold,
                                            target_df=df)
        
        logger.info(f"Exp:: Splited train and test data")


    def get_i_th_fold(self,i:int=0):
        #update split for train and valid set
        # pdb.set_trace()
        self.data_cohort.get_k_th_fold(df = self.data_cohort.data["train_val"],
                                        idx_lists=self.data_cohort.data["idxs"],
                                        test_data=self.data_cohort.data['test'],
                                        k_th_fold=i,
                                        )


    ################################################################
    #   setup experiment
    ################################################################ 
    def setup_experiment(self,main_data_source:str,need_train:bool=True,**kwargs):
        self.need_train = need_train
        if need_train:
            #------> for slide 
            if main_data_source == "slide":
                #-------train need split data
                label_idx = self.paras.cohort_para.targets[self.paras.cohort_para.targets_idx]
                self.data_cohort.show_taskcohort_stat(label_idx=label_idx)
                self.split_train_test()  # updated to split into train, valid, test
                # init train worker
                from HistoMIL.EXP.trainer.slide import pl_slide_trainer
                self.exp_worker = pl_slide_trainer(
                                        trainer_para =self.paras.trainer_para,
                                        dataset_para=self.paras.dataset_para,
                                        opt_para=self.paras.opt_para)
            else:
                raise NotImplementedError
            self.exp_worker.get_env_info(machine=self.machine,user=self.user,
                                        project=self.project,
                                        entity=self.entity,
                                        exp_name=self.exp_name)
            self.exp_worker.set_cohort(self.data_cohort)
            # pdb.set_trace()
            if self.cohort_para.in_domain_split_seed:
                self.exp_worker.get_in_domain_datapack(self.machine,self.paras.collector_para)
            else:
                self.exp_worker.get_datapack(self.machine,self.paras.collector_para)

            self.exp_worker.build_model()       # creates model from available implementations
            self.exp_worker.build_trainer()     # sets up trainer configurations such as wandb and learning rate
            # pdb.set_trace()
            # update paras
            self.paras.dataset_para=self.exp_worker.dataset_para
            self.paras.trainer_para=self.exp_worker.trainer_para
            self.paras.opt_para=self.exp_worker.opt_para
            # pdb.set_trace()
            #self.exp_worker.train()
        else:
            raise NotImplementedError
    
    def setup_cv_experiment(self,main_data_source:str,last_cv:int = 0, need_train:bool=True,**kwargs):
        self.need_train = need_train
        if main_data_source == "slide":
            #-------train need split data
            label_idx = self.paras.cohort_para.targets[self.paras.cohort_para.targets_idx] # coincidentally doesn't interfere with regression if i set targets-name to [g0_arrest, g0_arrest_score] because g0_arrest are the scores
            self.data_cohort.show_taskcohort_stat(label_idx=label_idx) 
            self.split_train_test()  # updated to split into train, valid, test
        
            for kfold in range(last_cv, self.paras.trainer_para.k_fold):
                
                if kfold != 0:
                    # logger.info(f"Resuming experiment from cross-validation fold {kfold} with \n checkpoint at: {self.paras.trainer_para.additional_pl_paras['resume_from_checkpoint']}")
                    logger.info(f"Resuming experiment from cross-validation fold {kfold}")
                
                
                #------> for slide 
                self.get_i_th_fold(i=kfold)
                # pdb.set_trace()
                # init train worker
                
                self.exp_worker = pl_slide_trainer(
                                        trainer_para =self.paras.trainer_para,
                                        dataset_para=self.paras.dataset_para,
                                        opt_para=self.paras.opt_para)
                
                self.exp_worker.get_env_info(machine=self.machine,user=self.user,
                                            project=self.project,
                                            entity=self.entity,
                                            exp_name=self.exp_name)
                self.exp_worker.set_cohort(self.data_cohort)
                # pdb.set_trace()
                if self.cohort_para.in_domain_split_seed:
                    self.exp_worker.get_in_domain_datapack(self.machine,self.paras.collector_para)
                else:
                    raise NotImplementedError
                    self.exp_worker.get_datapack(self.machine,self.paras.collector_para)

                self.exp_worker.build_model()       # creates model from available implementations
                self.exp_worker.trainer_para.ckpt_format = f'cv={kfold}' + "_{epoch:02d}-{auroc_val:.2f}"
                
                self.exp_worker.build_trainer(reinit=True)     # sets up trainer configurations such as wandb and learning rate
                # pdb.set_trace()
                # update paras
                self.paras.dataset_para=self.exp_worker.dataset_para
                self.paras.trainer_para=self.exp_worker.trainer_para
                self.paras.opt_para=self.exp_worker.opt_para
                # for clinical
                self.paras.dataset_para.current_fold = f'cv{kfold}'
                # pdb.set_trace()
                self.exp_worker.train()
                # pdb.set_trace()
                val_results = self.exp_worker.validate()

                print(val_results)

                ## restart pytorch lightning's configuration so that it doesn't load from previous checkpoint
                self.paras.trainer_para.additional_pl_paras = get_pl_trainer_additional_paras(self.paras.trainer_para.model_name)
                wandb.finish()
       
    def ensemble_test(self, ckpt_filenames, main_data_source:str = 'slide', ensemble:bool=True):
        
        mdl_ckpt_root = f"{self.paras.exp_locs.abs_loc('saved_models')}"
        
        if main_data_source == "slide":
            #-------train need split data
            label_idx = self.paras.cohort_para.targets[self.paras.cohort_para.targets_idx] # coincidentally doesn't interfere with regression if i set targets-name to [g0_arrest, g0_arrest_score] because g0_arrest are the scores
            self.data_cohort.show_taskcohort_stat(label_idx=label_idx) 
            self.split_train_test()  # updated to split into train, valid, test
            ensembled_probs = np.zeros((len(self.data_cohort.data['test'])))
            for ckpt in ckpt_filenames:
                
                
                #------> for slide 
                
                # pdb.set_trace()
                # init train worker
                # self.paras.trainer_para.additional_pl_paras.update(
                #     {'resume_from_checkpoint': f'{mdl_ckpt_root}{ckpt}.ckpt'}
                # )
                self.exp_worker = pl_slide_trainer(
                                        trainer_para =self.paras.trainer_para,
                                        dataset_para=self.paras.dataset_para,
                                        opt_para=self.paras.opt_para)
                
                self.exp_worker.get_env_info(machine=self.machine,user=self.user,
                                            project=self.project,
                                            entity=self.entity,
                                            exp_name=self.exp_name)
                self.exp_worker.set_cohort(self.data_cohort)
                # pdb.set_trace()
                if self.cohort_para.in_domain_split_seed:
                    self.exp_worker.get_in_domain_datapack(self.machine,self.paras.collector_para)
                else:
                    raise NotImplementedError
                    self.exp_worker.get_datapack(self.machine,self.paras.collector_para)

                self.exp_worker.build_model()       # creates model from available implementations
                
                
                self.exp_worker.build_inference_trainer(reinit=True)     # sets up trainer configurations such as wandb and learning rate
                # pdb.set_trace()
                # update paras
                self.paras.dataset_para=self.exp_worker.dataset_para
                self.paras.trainer_para=self.exp_worker.trainer_para
                self.paras.opt_para=self.exp_worker.opt_para
                
                # mdl = self.exp_worker.pl_model.load_from_checkpoint(f'{mdl_ckpt_root}{ckpt}.ckpt'})
                # print("first approach")
                # self.exp_worker.test() # but what does self.pl_model is equated to? it's equated to a initialized pytorch lightning model from SCRATCH
                # for clinical
                self.paras.dataset_para.current_fold = 'test'
                # print('second approach') # load from checkpoint EXPLICITLY
                best_cv_ckpt_path = f'{mdl_ckpt_root}{ckpt}.ckpt'
                # torch.load(best_cv_ckpt_path)['hyper_parameters']['additional_feature']
                self.exp_worker.pl_model = self.exp_worker.pl_model.load_from_checkpoint(best_cv_ckpt_path)
                # pdb.set_trace()
                self.exp_worker.test_from_checkpoint(model=self.exp_worker.pl_model)
                # pdb.set_trace()
                # self.exp_worker.data_pack['testset'].data

                cv_csv_loc = f"{self.paras.exp_locs.abs_loc('out_files')}{ckpt}.csv"
                self.exp_worker.pl_model.outputs.to_csv(cv_csv_loc, index=False)
                # pdb.set_trace()
                ensembled_probs += self.exp_worker.pl_model.outputs['probs'] if not ('Regression' in self.paras.trainer_para.model_name) else self.exp_worker.pl_model.outputs['prediction']
                # 'out_files'
                # pdb.set_trace()
                # ## restart pytorch lightning's configuration so that it doesn't load from previous checkpoint
                # self.paras.trainer_para.additional_pl_paras = get_pl_trainer_additional_paras(self.paras.trainer_para.model_name)
                wandb.finish()
            if ensemble:
                ensembled_probs /= len(ckpt_filenames)
                if 'Regression' in self.paras.trainer_para.model_name:
                    ensembled_binary_preds = (ensembled_probs <= self.exp_worker.pl_model.paras.threshold).astype(np.int8) # self.exp_worker.pl_model.binarize(ensembled_probs).astype(np.int8) 
                    binary_ground_truth = self.exp_worker.pl_model.outputs['binary_ground_truth'].astype(np.int8)
                    ground_truth = self.exp_worker.pl_model.outputs['continuous_ground_truth'].astype(np.float16)
                    correct = (ensembled_binary_preds == binary_ground_truth).astype(np.int8)
                    # pdb.set_trace()
                    ensembled_pcc = r2_score(ground_truth, ensembled_probs)
                    ensembled_f1 = f1_score(binary_ground_truth, ensembled_binary_preds)
                    ensembled_df = pd.DataFrame({
                        'ensemble_continuous_preds': ensembled_probs.values,
                        'continuous_ground_truth': ground_truth.values,
                        'ensemble_binary_preds': ensembled_binary_preds.values,
                        'binary_ground_truth': binary_ground_truth.values,
                        'binary_correct': correct
                    })
                    ensemble_loc = f"{self.paras.exp_locs.abs_loc('out_files')}ensemble_res_{self.paras.trainer_para.model_name}_{self.paras.collector_para.feature.model_name}.csv"
                    ensembled_df.to_csv(f'{ensemble_loc}', index=False)
                    print(f'ensemble test F1: {ensembled_f1}; ensemble test PCC {ensembled_pcc}')
                    # pdb.set_trace()
                else:
                    
                    ensembled_preds = (ensembled_probs >= 0.5).astype(np.int8) 
                    ground_truth = self.exp_worker.pl_model.outputs['ground_truth'].astype(np.int8)
                    correct = (ensembled_preds == ground_truth).astype(np.int8)
                    # pdb.set_trace()
                    ensembled_auroc = roc_auc_score(ground_truth, ensembled_probs)
                    ensembled_f1 = f1_score(ground_truth, ensembled_preds)
                    # Create the DataFrame
                    ensembled_df = pd.DataFrame({
                        'ensemble_probs': ensembled_probs.values,
                        'ensemble_preds': ensembled_preds.values,
                        'ground_truth': ground_truth.values,
                        'correct': correct
                    })
                    ensemble_loc = f"{self.paras.exp_locs.abs_loc('out_files')}ensemble_res_{self.paras.trainer_para.model_name}_{self.paras.collector_para.feature.model_name}.csv"
                    ensembled_df.to_csv(f'{ensemble_loc}', index=False)
                    print(f'ensemble test F1: {ensembled_f1}; ensemble test AUROC {ensembled_auroc}')



    def ensemble_heatmap(self, ckpt_filenames, main_data_source:str = 'slide', ensemble:bool=True):
        
        mdl_ckpt_root = f"{self.paras.exp_locs.abs_loc('saved_models')}"
        heatmap_task_root = f"{self.paras.data_locs.root}Heatmap/{self.paras.cohort_para.task_name}/{self.paras.collector_para.feature.model_name}/{self.paras.trainer_para.model_name}/"
        os.makedirs(heatmap_task_root, exist_ok=True)
        if main_data_source == "slide":
            #-------train need split data
            label_idx = self.paras.cohort_para.targets[self.paras.cohort_para.targets_idx] # coincidentally doesn't interfere with regression if i set targets-name to [g0_arrest, g0_arrest_score] because g0_arrest are the scores
            self.data_cohort.show_taskcohort_stat(label_idx=label_idx) 
            self.split_train_test()  # updated to split into train, valid, test
            ensembled_probs = np.zeros((len(self.data_cohort.data['test'])))
            
                
            self.exp_worker = pl_slide_trainer(
                                    trainer_para =self.paras.trainer_para,
                                    dataset_para=self.paras.dataset_para,
                                    opt_para=self.paras.opt_para)
            
            self.exp_worker.get_env_info(machine=self.machine,user=self.user,
                                        project=self.project,
                                        entity=self.entity,
                                        exp_name=self.exp_name)
            self.exp_worker.set_cohort(self.data_cohort)
            # pdb.set_trace()
            if self.cohort_para.in_domain_split_seed:
                self.exp_worker.get_in_domain_datapack(self.machine,self.paras.collector_para)
            else:
                raise NotImplementedError
                
            self.exp_worker.build_model()       # creates model from available implementations
            self.paras.trainer_para.with_logger = None # disable wandb for heatmap generation
            self.exp_worker.build_inference_trainer(reinit=True)     # sets up trainer configurations such as wandb and learning rate
            # pdb.set_trace()
            # update paras
            self.paras.dataset_para=self.exp_worker.dataset_para
            self.paras.trainer_para=self.exp_worker.trainer_para
            self.paras.opt_para=self.exp_worker.opt_para
            
            self.paras.dataset_para.current_fold = 'test'
            testloader = self.exp_worker.data_pack['testloader']
            testdataset = self.data_cohort.data['test']
            
            for idx, batch in enumerate(testloader):
                ### get paths for wsi .svs file, segmented tissue and patch coords of segmented tissue  
                patient_id, folder, filename, label = testdataset.iloc[idx][['PatientID', 'folder', 'filename', self.paras.cohort_para.task_name]]
                
                wsi_path = f"{self.paras.data_locs.abs_loc('slide')}{folder}/{filename}"
                wsi_tissue_path = f"{self.paras.data_locs.abs_loc('tissue')}{folder}.{filename}.pkl"
                wsi_coords_path = f"{self.paras.data_locs.abs_loc('patch')}{self.paras.collector_para.patch.step_size}_{self.paras.collector_para.patch.step_size}/{folder}.{filename}.h5"
                # pdb.set_trace()
                ### read svs file, along with tissue and patch
                wsi_object = WholeSlideImageHeatmap(path=wsi_path)
                with open(wsi_tissue_path, 'rb') as f:
                    wsi_tissue = pickle.load(f)
                wsi_coords = h5py.File(wsi_coords_path)
                wsi_coords = wsi_coords['coords']
                wsi_object.contours_tissue = wsi_tissue['tissue']
                wsi_object.holes_tissue = wsi_tissue['holes']

                ### compute and process attention scores
                ensemble_attn_scores = []
                ensemble_probs = []
                for ckpt in ckpt_filenames:
                    best_cv_ckpt_path = f'{mdl_ckpt_root}{ckpt}.ckpt'
                
                    self.exp_worker.pl_model = self.exp_worker.pl_model.load_from_checkpoint(best_cv_ckpt_path)

                    logits, Y_prob, Y_hat, A = self.exp_worker.pl_model.infer_step(batch)
                    
                    if A.dim() == 3:
                        A = A.mean(-1) # aggregate attention vectors
                    if A.shape[-1] > len(wsi_coords):
                        
                        print('# of patches:', len(wsi_coords))
                        print('# of attn scores exceeded:', A.shape)
                        # pdb.set_trace()
                        A = A[:, :len(wsi_coords)]

                        
                    # Y_hat = Y_hat.item()
                    # probs, ids = torch.topk(Y_prob, 1)
                    # probs = probs[-1].cpu().numpy()
                    # ids = ids[-1].cpu().numpy()
                    Y_hats, Y_probs, A =  Y_hat.item(), Y_prob.item(),  A.view(-1, 1).cpu().numpy() 
                    ensemble_attn_scores.append(A)
                    ensemble_probs.append(Y_probs)
                averaged_attn_scores = np.mean(ensemble_attn_scores, axis=0)
                averaged_probs = np.mean(ensemble_probs, axis=0)
                if 'Regression' in self.paras.trainer_para.model_name:
                    label = round(label, 2)
                    Y_hats = np.round(logits.numpy()[0], 2)
                # pdb.set_trace()
                    
                ### heatmap and patch args
                patch_size = self.paras.collector_para.patch.patch_size
                patch_level = 0
                wsi_ref_downsample = wsi_object.level_downsamples[0]
                patch_custom_downsample = 1
                vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_custom_downsample ).astype(int))
                overlap = 0
                top_left = None
                bot_right = None
                use_roi = False
                heatmap_vis_args = {'convert_to_percentiles': True, 'blur': True, 'custom_downsample': 1}
                use_ref_scores = True
                blank_canvas = False
                alpha = 0.36
                vis_level = 1
                binarize = False
                binary_thresh = -1
                cmap = 'jet'
                save_ext = 'jpg'
                tag = "label_{}_pred_{:.2f}".format(label, Y_hats) if 'Regression' in self.paras.trainer_para.model_name else "label_{}_pred_{}".format(label, Y_hats)
                tag = f"ensemble_{tag}" if ensemble else tag 
                if ensemble:
                    A = averaged_attn_scores
                    heatmap_save_name = 'ensemble_{}_{}_o_{}_roi_{}_blur_{}_refs_{}_bc_{}_alpha_{}_visl_{}_bi_{}_{}.{}'.format(patient_id, tag, overlap, int(use_roi),
                                                                                    int(heatmap_vis_args['blur']), 
                                                                                    int(use_ref_scores), int(blank_canvas), 
                                                                                    float(alpha), int(vis_level), 
                                                                                    int(binarize), float(binary_thresh), save_ext)
                else:
                    heatmap_save_name = '{}_{}_o_{}_roi_{}_blur_{}_refs_{}_bc_{}_alpha_{}_visl_{}_bi_{}_{}.{}'.format(patient_id, tag, overlap, int(use_roi),
                                                                                    int(heatmap_vis_args['blur']), 
                                                                                    int(use_ref_scores), int(blank_canvas), 
                                                                                    float(alpha), int(vis_level), 
                                                                                    int(binarize), float(binary_thresh), save_ext)
                ### sampling patches for close examination
                
                samples = [{'name': 'topk_high_attention', 'sample': True, 'seed': 42, 'k': 15, 'mode': 'topk'}, {'name': 'reverse_topk_high_attention', 'sample': True, 'seed': 42, 'k': 15, 'mode': 'reverse_topk'}]


                for sample in samples:
                    if sample['sample']:
                        
                        sample_save_dir =  os.path.join(heatmap_task_root, 'sampled_patches', str(tag), sample['name'])
                        os.makedirs(sample_save_dir, exist_ok=True)
                        print('sampling {}'.format(sample['name']))
                        sample_results = sample_rois(scores=A, coords=wsi_coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                            score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                        for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                            print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                            patch = wsi_object.wsi.read_region(tuple(s_coord), patch_level, patch_size).convert('RGB')
                            patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, patient_id, s_coord[0], s_coord[1], s_score)))
                
                ### generating heatmap
                if os.path.isfile(os.path.join(heatmap_task_root, heatmap_save_name)):
                    print("Heatmap already computed")
                    pass
                else:
                    heatmap = wsi_object.visHeatmap(scores=A, coords=wsi_coords, vis_level=vis_level,  
                                cmap=cmap, alpha=alpha, **heatmap_vis_args, 
                                binarize=binarize, blank_canvas=blank_canvas,
                                thresh=binary_thresh,  patch_size = vis_patch_size,
                                overlap=overlap, top_left=top_left, bot_right = bot_right)
                    
                    heatmap.save(os.path.join(heatmap_task_root, heatmap_save_name), quality=100)

            
                    
    def run(self):
        if self.need_train:
            self.exp_worker.train()
        else:
            raise NotImplementedError
            #self.exp_worker.save()

    ################################################################
    #   ssl experiment
    ################################################################ 




        



        
