"""
pytorch-lightning wrapper for the DTFD-MODEL
"""

#---->
import pytorch_lightning as pl
import pandas as pd
import pytorch_lightning as pl

import seaborn as sns
import torch
import torchmetrics
import wandb
from matplotlib import pyplot as plt
#---->
from HistoMIL import logger
# from HistoMIL.MODEL.Image.PL_protocol.MIL import pl_MIL
# from HistoMIL.EXP.paras.dataset import DatasetParas
# from HistoMIL.EXP.paras.optloss import OptLossParas
# from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.MODEL.Image.MIL.DTFDTransMIL.paras import DTFDTransMILParas
from HistoMIL.MODEL.Image.MIL.DTFDTransMIL.model import DTFDTransMIL
from HistoMIL.MODEL.Image.MIL.utils import  get_loss, get_optimizer, get_scheduler

from pytorch_lightning.utilities import rank_zero_only # for saving a single model in a multi-gpu setting
import os
import pdb
#---->
####################################################################################
#      pl protocol class
####################################################################################

class pl_DTFDTransMIL(pl.LightningModule):
    def __init__(self, paras:DTFDTransMILParas):
        super().__init__()
        self.automatic_optimization = False
        self.paras = paras
        self.model = DTFDTransMIL(paras)# 

        self.criterion = get_loss(paras.criterion
                                 ) if paras.task == "binary" else get_loss(paras.criterion)
        self.save_hyperparameters()

        self.lr = paras.lr
        self.weight_decay = paras.weight_decay

        self.acc_train = torchmetrics.Accuracy(task=paras.task, num_classes=paras.num_cls)
        self.acc_val = torchmetrics.Accuracy(task=paras.task, num_classes=paras.num_cls)
        self.acc_test = torchmetrics.Accuracy(task=paras.task, num_classes=paras.num_cls)

        self.auroc_val = torchmetrics.AUROC(
            task=paras.task,
            num_classes=paras.num_cls,
        )
        self.auroc_test = torchmetrics.AUROC(
            task=paras.task,
            num_classes=paras.num_cls,
        )

        self.f1_val = torchmetrics.F1Score(
            task=paras.task,
            num_classes=paras.num_cls,
        )
        self.f1_test = torchmetrics.F1Score(
            task=paras.task,
            num_classes=paras.num_cls,
        )

        self.precision_val = torchmetrics.Precision(
            task=paras.task,
            num_classes=paras.num_cls,
        )
        self.precision_test = torchmetrics.Precision(
            task=paras.task,
            num_classes=paras.num_cls,
        )

        self.recall_val = torchmetrics.Recall(
            task=paras.task,
            num_classes=paras.num_cls,
        )
        self.recall_test = torchmetrics.Recall(
            task=paras.task,
            num_classes=paras.num_cls,
        )

        self.specificity_val = torchmetrics.Specificity(
            task=paras.task,
            num_classes=paras.num_cls,
        )
        self.specificity_test = torchmetrics.Specificity(
            task=paras.task,
            num_classes=paras.num_cls,
        )

        self.cm_val = torchmetrics.ConfusionMatrix(task=paras.task, num_classes=paras.num_cls)
        self.cm_test = torchmetrics.ConfusionMatrix(
            task=paras.task, num_classes=paras.num_cls
        )
        # For gradient accumulation
        self.accumulation_steps = 8
        self.accumulation_count = 0

        # for manual model checkpointing
        self.best_auroc_val = 0

    def forward(self, x):
        slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = self.model(x)
        # pdb.set_trace()
        return slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred

    def configure_optimizers(self):
        trainable_parameters = list(self.model.TransMIL.parameters())

        optimizer0 = torch.optim.AdamW(trainable_parameters, lr=self.paras.lr, weight_decay=self.paras.weight_decay)
        optimizer1 = torch.optim.Adam(self.model.attCls.parameters(), lr=self.paras.lr, weight_decay=self.paras.weight_decay)
       
        scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=42, eta_min=1e-6)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [25], gamma=self.paras.lr_decay_ratio)
        
        return [optimizer0, optimizer1], [scheduler0, scheduler1]
   

    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        # Get the optimizers
        opt0, opt1 = self.optimizers()
        
        # Enable gradient computation if it's not already enabled
        torch.set_grad_enabled(True)
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast():
            inputs, labels = batch  # 
            
            slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = self.forward(batch)
        
            # pdb.set_trace()
            labels = labels.unsqueeze(1)
            slide_sub_labels = slide_sub_labels.unsqueeze(1)
            
            ### first loss optimization and logging
            loss0 = self.criterion(slide_sub_preds, slide_sub_labels.float()).mean()
            ### second loss optimization and logging
            loss1 = self.criterion(gSlidePred, labels.float()).mean()
        # Gradient accumulation for the first optimizer
        self.manual_backward(loss0 / self.accumulation_steps, retain_graph=True)
        
        # Gradient accumulation for the second optimizer
        self.manual_backward(loss1 / self.accumulation_steps, retain_graph=True)
        
        self.accumulation_count += 1
        
        # Perform optimization step if we've accumulated enough gradients
        if self.accumulation_count == self.accumulation_steps:
            # Clip gradients
            self.clip_gradients(opt0, gradient_clip_val=self.paras.grad_clipping, gradient_clip_algorithm="norm")
            self.clip_gradients(opt1, gradient_clip_val=self.paras.grad_clipping, gradient_clip_algorithm="norm")
            
            # Step optimizers
            opt0.step()
            opt1.step()
            
            # Zero gradients
            opt0.zero_grad()
            opt1.zero_grad()
            
            # Reset accumulation count
            self.accumulation_count = 0
        
        # Log losses
        self.log('train_loss_0', loss0, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_1', loss1, on_step=True, on_epoch=True, prog_bar=True)
        # Log training acc
        probs = torch.sigmoid(slide_sub_preds)
        preds = torch.round(probs)
        self.acc_train(preds, slide_sub_labels.float())
        
        return {"loss0": loss0, "loss1": loss1}
        

    def validation_step(self, batch, batch_idx):
        self.model.TransMIL.eval()
        self.model.attCls.eval()
        # pdb.set_trace()

        # pdb.set_trace()
        inputs, labels = batch  # 
        with torch.no_grad():
            slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = self.forward(batch)


        labels = labels.unsqueeze(1)
        # pdb.set_trace()
        loss = self.criterion(gSlidePred, labels.float()).mean()
        # print(y)
        probs = torch.sigmoid(gSlidePred)
        preds = torch.round(probs)
         
        # the model's predictions is glidepred
        self.acc_val(probs, labels)
        self.auroc_val(probs, labels)
        self.f1_val(probs, labels)
        self.precision_val(probs, labels)
        self.recall_val(probs, labels)
        self.specificity_val(probs, labels)
        # pdb.set_trace()
        self.cm_val(probs, labels)

        self.log("loss_val", loss, prog_bar=True)
        self.log("acc_val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc_val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1_val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_val", self.precision_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall_val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "specificity_val", self.specificity_val, prog_bar=False, on_step=False, on_epoch=True
        )
    @rank_zero_only
    def save_best_model(self, auroc_val):
        if isinstance(auroc_val, torch.Tensor):
            auroc_val = auroc_val.item()
        elif hasattr(auroc_val, 'compute'):
            auroc_val = auroc_val.compute().item()
        # pdb.set_trace()
        if auroc_val >= self.best_auroc_val:
            self.best_auroc_val = auroc_val
            epoch = self.trainer.current_epoch
            # pdb.set_trace()
            filename = self.trainer.checkpoint_callback.filename # .format() + f"{epoch:02d}-{self.best_auroc_val:.2f}.ckpt"
            filename = filename.format(epoch=epoch, auroc_val=self.best_auroc_val) + ".ckpt"
            filepath = os.path.join(self.trainer.checkpoint_callback.dirpath, \
                                    filename)
            self.trainer.save_checkpoint(filepath)
            print(f"Saved new best model: {filepath}")
            
            # Update the best model path in the ModelCheckpoint callback
            self.trainer.checkpoint_callback.best_model_path = filepath
            # pdb.set_trace()   
            self.trainer.checkpoint_callback.best_model_score = torch.tensor(auroc_val)
    def on_validation_epoch_end(self):
        if self.global_step != 0:
            cm = self.cm_val.compute()

            # normalise the confusion matrix
            norm = cm.sum(axis=1, keepdims=True)
            normalized_cm = cm / norm

            # log to wandb
            plt.clf()
            cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
            wandb.log({"confusion_matrix_val": wandb.Image(cm)})

        self.cm_val.reset()
        self.save_best_model(self.auroc_val)

    def on_test_epoch_start(self) -> None:
        # save test outputs in dataframe per test dataset
        column_names = ['patient', 'ground_truth', 'predictions', 'logits', 'correct']
        self.outputs = pd.DataFrame(columns=column_names)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.TransMIL.eval()
        self.model.attCls.eval()
        # pdb.set_trace()
        inputs, labels = batch  # 
        with torch.no_grad():
            slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = self.forward(batch)


        labels = labels.unsqueeze(1)

        
        
        loss = self.criterion(gSlidePred, labels).mean()

        probs = torch.sigmoid(gSlidePred)
        preds = torch.round(probs)
        
        self.acc_test(probs, labels)
        self.auroc_test(probs, labels)
        self.f1_test(probs, labels)
        self.precision_test(probs, labels)
        self.recall_test(probs, labels)
        self.specificity_test(probs, labels)
        self.cm_test(probs, labels)

        self.log("loss_test", loss, prog_bar=False)
        self.log("acc_test", self.acc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc_test", self.auroc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1_test", self.f1_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "precision_test", self.precision_test, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log("recall_test", self.recall_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "specificity_test", self.specificity_test, prog_bar=False, on_step=False, on_epoch=True
        )

        outputs = pd.DataFrame(
            data=[
                [
                 labels.item(),
                 preds.item(),
                 gSlidePred.squeeze(), (labels == preds).int().item()]
            ],
            columns=[ 'ground_truth', 'prediction', 'logits', 'correct']
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)

    def on_test_epoch_end(self):
        if self.global_step != 0:
            cm = self.cm_test.compute()

            # normalise the confusion matrix
            norm = cm.sum(axis=1, keepdims=True)
            normalized_cm = cm / norm

            # log to wandb
            plt.clf()
            cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
            wandb.log({"confusion_matrix_test": wandb.Image(cm)})

        self.cm_test.reset()

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

