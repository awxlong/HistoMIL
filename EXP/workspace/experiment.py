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

import wandb

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
            label_idx = self.paras.cohort_para.targets[self.paras.cohort_para.targets_idx]
            self.data_cohort.show_taskcohort_stat(label_idx=label_idx)
            self.split_train_test()  # updated to split into train, valid, test
        
            for kfold in range(last_cv, self.paras.trainer_para.k_fold):
                
                if kfold != 0:
                    logger.info(f"Resuming experiment from cross-validation fold {kfold} with \n checkpoint at: {self.paras.trainer_para.additional_pl_paras['resume_from_checkpoint']}")
                
                
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
                # pdb.set_trace()
                self.exp_worker.train()
                val_results = self.exp_worker.validate()

                print(val_results)
                ## restart pytorch lightning's configuration so that it doesn't load from previous checkpoint
                
                self.paras.trainer_para.additional_pl_paras = {
                                                            #---------> paras for pytorch lightning trainner
                                                            "accumulate_grad_batches":8, # mil need accumulated grad
                                                            "accelerator":"auto",        #accelerator='gpu', devices=1,
                                                            'precision': 16,             # Use mixed precision
                                                            'enable_progress_bar': True, 
                                                            'enable_model_summary': True,} 
                wandb.finish()
    # def setup_cv_experiment_rerun(self, main_data_source:str, last_cv:int = 0, need_train:bool=True):
    #     '''
    #     TODO: Refactor this by editing 
    #     '''
    #     self.need_train = need_train
    #     if main_data_source == "slide":
    #         #-------train need split data
    #         label_idx = self.paras.cohort_para.targets[self.paras.cohort_para.targets_idx]
    #         self.data_cohort.show_taskcohort_stat(label_idx=label_idx)
    #         self.split_train_test()  # updated to split into train, valid, test
        
    #         for kfold in range(last_cv, self.paras.trainer_para.k_fold):

    #             #------> for slide 
    #             self.get_i_th_fold(i=kfold)
    #             # init train worker
                
    #             self.exp_worker = pl_slide_trainer(
    #                                     trainer_para =self.paras.trainer_para,
    #                                     dataset_para=self.paras.dataset_para,
    #                                     opt_para=self.paras.opt_para)
                
    #             self.exp_worker.get_env_info(machine=self.machine,user=self.user,
    #                                         project=self.project,
    #                                         entity=self.entity,
    #                                         exp_name=self.exp_name)
    #             self.exp_worker.set_cohort(self.data_cohort)
    #             # pdb.set_trace()
    #             if self.cohort_para.in_domain_split_seed:
    #                 self.exp_worker.get_in_domain_datapack(self.machine,self.paras.collector_para)
    #             else:
    #                 raise NotImplementedError
    #                 self.exp_worker.get_datapack(self.machine,self.paras.collector_para)

    #             self.exp_worker.build_model()       # creates model from available implementations
    #             self.exp_worker.trainer_para.ckpt_format = f'cv={kfold}' + "_{epoch:02d}-{auroc_val:.2f}"
                
    #             self.exp_worker.build_trainer(reinit=True)     # sets up trainer configurations such as wandb and learning rate
    #             # pdb.set_trace()
    #             # update paras
    #             self.paras.dataset_para=self.exp_worker.dataset_para
    #             self.paras.trainer_para=self.exp_worker.trainer_para
    #             self.paras.opt_para=self.exp_worker.opt_para
    #             # pdb.set_trace()
    #             self.exp_worker.train()
    #             val_results = self.exp_worker.validate()

    #             print(val_results)
    #             wandb.finish()        
        
    def run(self):
        if self.need_train:
            self.exp_worker.train()
        else:
            raise NotImplementedError
            #self.exp_worker.save()

    ################################################################
    #   ssl experiment
    ################################################################ 




        



        
