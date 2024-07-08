import os
import numpy as np
import pytorch_lightning as pl

import torch
from pathlib import Path

from HistoMIL import logger
from HistoMIL.EXP.workspace.env import Machine
from HistoMIL.EXP.paras.slides import CollectorParas
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.DATA.Cohort.data import DataCohort
from HistoMIL.DATA.Database.dataset import create_slide_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pdb
class pl_base_trainer:
    """
    base class for pl trainer pipeline
    """
    def __init__(self,
                trainer_para:PLTrainerParas,
                dataset_para:DatasetParas,
                opt_para:OptLossParas,):
        
        self.trainer_para = trainer_para 
        self.dataset_para = dataset_para
        self.opt_para = opt_para

        self.data_pack = {}
        self.pl_model = None

        self.machine = None
        self.user = None
        self.project = None
        self.entity = None
        self.exp_name = None

    ################################################################
    #   build common function trainner
    ################################################################ 
    def get_env_info(self,machine,user,project,entity,exp_name):
        self.machine = machine
        self.user = user
        self.project = project
        self.entity = entity
        self.exp_name = exp_name
        
    def build_trainer(self, reinit=False):
        trainer_additional_dict = self.trainer_para.additional_pl_paras
        callbacks_list = []

        # 4. create learning rate logger
        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks_list.append(lr_monitor)
        if self.trainer_para.with_logger=="wandb":
            # 4. Create wandb logger
            from pytorch_lightning.loggers import WandbLogger
            os.environ["WANDB_API_KEY"]=self.user.wandb_api_key

            wandb_logger = WandbLogger(project=self.project,
                                        entity=self.entity,
                                        name=self.exp_name,
                                        reinit=reinit)
            # pdb.set_trace()
            trainer_additional_dict.update({"logger":wandb_logger})

        if self.trainer_para.with_ckpt:
            # 4. check point
            
            ckpt_paras = self.trainer_para.ckpt_para
            ckpt_name = self.exp_name+self.trainer_para.ckpt_format
            # pdb.set_trace()
            logger.debug(f"Trainer:: for exp {self.exp_name} Checkpoint with paras {ckpt_paras}")
            checkpoint_callback = ModelCheckpoint(dirpath=self.machine.exp_locs.abs_loc("saved_models"),
                                                filename=ckpt_name,
                                                 **ckpt_paras,
                                                )
            ckpt_dir = self.machine.exp_locs.abs_loc("saved_models")
            logger.info(f"Trainer:: Best model will be saved at {ckpt_dir} as {ckpt_name}")
            callbacks_list.append(checkpoint_callback)
            
        if len(callbacks_list)>=1: trainer_additional_dict.update(
                                            {"callbacks":callbacks_list})
        # pdb.set_trace()
        # 4. Trainer and fit
        self.trainer = pl.Trainer(default_root_dir=self.machine.exp_locs.\
                                                        abs_loc("out_files"),
                                max_epochs=self.opt_para.max_epochs,
                                **trainer_additional_dict
                                )
        

    def train(self):
        logger.info("Trainer:: Start training....")
        trainloader = self.data_pack["trainloader"] 
        validloader = self.data_pack["validloader"]


        self.trainer.fit(model=self.pl_model, 
                train_dataloaders=trainloader,
                val_dataloaders=validloader)
        
    def train_val_kfold(self):
        logger.info("Trainer:: Start training in k-fold cross validation....")
        
        trainloader = self.data_pack["trainloader"] 
        validloader = self.data_pack["validloader"]


        self.trainer.fit(model=self.pl_model, 
                train_dataloaders=trainloader,
                val_dataloaders=validloader)

    def validate(self,ckpt_path:str="best"):
        validloader = self.data_pack["validloader"]
        out = self.trainer.validate(dataloaders=validloader, ckpt_path=ckpt_path,)
        return out
    
    def test(self, ckpt_path:str='best'):
        test_loader = self.data_pack['testloader']
        out = self.trainer.test(dataloaders=test_loader, ckpt_path=ckpt_path,)
        return out 
    
    ################################################################
    #   cohort related function
    ################################################################ 
    def set_cohort(self,data_cohort:DataCohort):
        self.data_cohort = data_cohort

    def in_domain_dataloader_init_fn_(self,train_phase:str,
                            machine:Machine,
                            collector_para:CollectorParas,):
        
        """
        produce dataset and dataloader for different training phase
        """
        slide_list,patch_list,_ = self.data_cohort.get_task_datalist(phase=train_phase)

        # different slide methods need different training protocol 
        # data_list related with training methods,
        # (1)mil need slide list 
        # (2)and transfer learning need patch list
        method_type = self.trainer_para.method_type# "mil" or "patch_learning"
        assert method_type in ["mil","patch_learning"]
        data_list = slide_list if method_type == "mil" else patch_list

        # is_train related with two things: dataset_para and train_phase
        is_train = self.dataset_para.is_train if self.dataset_para.is_train is not None \
                                         else True if train_phase == "train" else False
        # get dataset
        dataset = create_slide_dataset(
                                data_list=data_list,
                                data_locs=machine.data_locs,
                                concept_paras=collector_para,
                                dataset_paras=self.dataset_para,
                                is_train=is_train,
                                as_PIL=self.dataset_para.as_PIL,
                                )

        return dataset,self.data_cohort.create_dataloader(dataset,
                                                    self.dataset_para)
    
        

    def change_label_dict(self,dataset,dataloader):
        # get original label dict
        label_dict = self.data_cohort.cohort_para.label_dict
        # get one example label and transfer to np array
        l_example = list(label_dict.keys())[0]
        if type(l_example) != list:l_example = [l_example] # make sure it is a list otherwise it will not have shape
        label_example = np.array(l_example) # convert to np array
        # pdb.set_trace()
        # check target format
        target_format = self.trainer_para.label_format

        # transfer label dict
        from HistoMIL.MODEL.Image.PL_protocol.utils import current_label_format,label_format_transfer
        current_format = current_label_format(label_example,task=self.trainer_para.task_type)
        if current_format!=target_format:
            target_dict = label_format_transfer(target_format,label_dict)
            dataset.label_dict = target_dict
            dataloader.dataset.label_dict = target_dict
            logger.info(f"Trainer:: Change label into: {target_dict}")


        return dataset,dataloader

    def get_datapack(self,
                    machine:Machine,
                    collector_para:CollectorParas,):
        """
        create self.datapack which include trainset,testset,
        trainloader, testloader
        in:
            machine:Machine: machine object for data path
            collector_para:CollectorParas: paras for data collector

        """
        is_shuffle = self.dataset_para.is_shuffle
        is_weight_sampler = self.dataset_para.is_weight_sampler
        #---> for train phase
        trainset,trainloader = self.dataloader_init_fn(train_phase="train",
                                            machine=machine,
                                            collector_para=collector_para)

        #---> for validation phase
        if not self.dataset_para.force_balance_val:
            self.dataset_para.is_shuffle=False # not shuffle for validation
            self.dataset_para.is_weight_sampler=False
        testset,testloader = self.dataloader_init_fn(train_phase="test",
                                            machine=machine,
                                            collector_para=collector_para)
        # pdb.set_trace()
        #---> setup dataset meta
        self.dataset_para.data_len = trainset.__len__()
        _,dict_L = trainset.get_balanced_weight(device="cpu")
        self.dataset_para.category_ratio = dict_L

        #----> change back for next run
        self.dataset_para.is_shuffle=is_shuffle # not shuffle for validation
        self.dataset_para.is_weight_sampler=is_weight_sampler
        
        #----> change label_dict to fit the model and loss
        # get original label dict
        trainset,trainloader = self.change_label_dict(trainset,trainloader)
        testset,testloader = self.change_label_dict(testset,testloader)
        #----> save to self
        self.data_pack = {
            "trainset":trainset,
            "trainloader":trainloader,
            "testset":testset,
            "testloader":testloader,
        }

    def get_in_domain_datapack(self,
                    machine:Machine,
                    collector_para:CollectorParas,):
        """
        create self.datapack which includes an in-domain trainset, validset, testset,
        trainloader, validloader, testloader
        in:
            machine:Machine: machine object for data path
            collector_para:CollectorParas: paras for data collector

        """
        is_shuffle = self.dataset_para.is_shuffle
        is_weight_sampler = self.dataset_para.is_weight_sampler
        #---> for train phase
        
        trainset,trainloader = self.in_domain_dataloader_init_fn_(train_phase="train",
                                            machine=machine,
                                            collector_para=collector_para)

        
        #---> for validation phase
        if not self.dataset_para.force_balance_val:
            self.dataset_para.is_shuffle=False # not shuffle for validation
            self.dataset_para.is_weight_sampler=False
        
        validset,validloader = self.in_domain_dataloader_init_fn_(train_phase="valid",
                                            machine=machine,
                                            collector_para=collector_para)
        testset,testloader = self.in_domain_dataloader_init_fn_(train_phase="test",
                                            machine=machine,
                                            collector_para=collector_para)
        # pdb.set_trace()
        #---> setup dataset meta
        self.dataset_para.data_len = trainset.__len__()
        _,dict_L = trainset.get_balanced_weight(device="cpu")
        self.dataset_para.category_ratio = dict_L

        #----> change back for next run
        self.dataset_para.is_shuffle=is_shuffle # not shuffle for validation
        self.dataset_para.is_weight_sampler=is_weight_sampler
        
        #----> change label_dict to fit the model and loss
        # get original label dict
        trainset,trainloader = self.change_label_dict(trainset, trainloader)
        validset,validloader = self.change_label_dict(validset, validloader)
        testset,testloader = self.change_label_dict(testset, testloader)
        #----> save to self
        # pdb.set_trace()
        self.data_pack = {
            "trainset":trainset,
            "trainloader":trainloader,
            "validset": validset,
            "validloader": validloader,
            "testset":testset,
            "testloader":testloader,
        }
