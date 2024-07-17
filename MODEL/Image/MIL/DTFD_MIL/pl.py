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

from HistoMIL.MODEL.Image.MIL.DTFD_MIL.paras import DTFD_MILParas
from HistoMIL.MODEL.Image.MIL.DTFD_MIL.model import DTFD_MIL
from HistoMIL.MODEL.Image.MIL.utils import  get_loss, get_optimizer, get_scheduler

#---->
####################################################################################
#      pl protocol class
####################################################################################

class pl_DTFDMIL(pl.LightningModule):
    def __init__(self, paras:DTFD_MILParas):
        super().__init__()
        self.automatic_optimization = False
        self.paras = paras
        self.model = DTFD_MIL(paras)
        self.criterion = get_loss(paras.criterion
                                 ) if paras.task == "binary" else get_loss(paras.criterion)
        self.save_hyperparameters()

        self.lr = paras.lr
        self.wd = paras.wd

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

    def forward(self, x):
        slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = self.model(x)
        return slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred

    def configure_optimizers(self):
        trainable_parameters = list(self.model.classifier.parameters()) + \
                               list(self.model.attention.parameters()) + \
                               list(self.model.dimReduction.parameters())

        optimizer0 = torch.optim.Adam(trainable_parameters, lr=self.paras.lr, weight_decay=self.paras.weight_decay)
        optimizer1 = torch.optim.Adam(self.attCls.parameters(), lr=self.paras.lr, weight_decay=self.paras.weight_decay)

        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, [100], gamma=self.paras.lr_decay_ratio)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [100], gamma=self.paras.lr_decay_ratio)

        return [optimizer0, optimizer1], [scheduler0, scheduler1]
   

    def training_step(self, batch, batch_idx, optimizer_idx):
        # pdb.set_trace()
        # Get the optimizers
        opt0, opt1 = self.optimizers()
        
        # Enable gradient computation if it's not already enabled
        torch.set_grad_enabled(True)
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast():
            inputs, labels = batch  # 
        
            slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = self.forward(batch)
        
        
            labels = labels.unsqueeze(1)
            slide_sub_labels = slide_sub_labels.unsqueeze(1)
            
            ### first loss optimization and logging
            loss0 = self.criterion(slide_sub_preds, slide_sub_labels.float()).mean()
            ### second loss optimization and logging
            loss1 = self.criterion(gSlidePred, labels).mean()
        # Gradient accumulation for the first optimizer
        self.manual_backward(loss0 / self.accumulation_steps)
        
        # Gradient accumulation for the second optimizer
        self.manual_backward(loss1 / self.accumulation_steps)
        
        self.accumulation_count += 1
        
        # Perform optimization step if we've accumulated enough gradients
        if self.accumulation_count == self.accumulation_steps:
            # Clip gradients
            self.clip_gradients(opt0, gradient_clip_val=self.params.grad_clipping, gradient_clip_algorithm="norm")
            self.clip_gradients(opt1, gradient_clip_val=self.params.grad_clipping, gradient_clip_algorithm="norm")
            
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
        self.acc_train(preds, slide_sub_labels.unsqueeze(1).float())
        
        return {"loss0": loss0, "loss1": loss1}
        

    def validation_step(self, batch, batch_idx):
        self.model.classifier.eval()
        self.model.dimReduction.eval()
        self.model.attention.eval()
        self.model.attCls.eval()
        # pdb.set_trace()
        inputs, labels = batch  # 
        with torch.no_grad():
            slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = self.forward(batch)


        labels = labels.unsqueeze(1)
        slide_sub_labels = slide_sub_labels.unsqueeze(1)
        
        ### first loss optimization and logging
        loss0 = self.criterion(slide_sub_preds, slide_sub_labels.float()).mean()
        ### second loss optimization and logging
        loss1 = self.criterion(gSlidePred, labels).mean()
        # print(y)
        # print(y.shape)
        
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
   
        # print(y)
        # print(probs.shape)
        # print(y.shape)
        self.acc_val(probs, y)
        self.auroc_val(probs, y)
        self.f1_val(probs, y)
        self.precision_val(probs, y)
        self.recall_val(probs, y)
        self.specificity_val(probs, y)
        # pdb.set_trace()
        self.cm_val(probs, y)

        self.log("loss_val", loss, prog_bar=True)
        self.log("acc_val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc_val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1_val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_val", self.precision_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall_val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "specificity_val", self.specificity_val, prog_bar=False, on_step=False, on_epoch=True
        )

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

    def on_test_epoch_start(self) -> None:
        # save test outputs in dataframe per test dataset
        column_names = ['patient', 'ground_truth', 'predictions', 'logits', 'correct']
        self.outputs = pd.DataFrame(columns=column_names)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch  # x = features, coords, y = labels, tiles, patient
        # pdb.set_trace()
        logits = self.forward(x)

        if self.paras.task == "binary":
            y = y.unsqueeze(1)
            loss = self.criterion(logits, y.float())
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)
        else:
            loss = self.criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True)

        self.acc_test(probs, y)
        self.auroc_test(probs, y)
        self.f1_test(probs, y)
        self.precision_test(probs, y)
        self.recall_test(probs, y)
        self.specificity_test(probs, y)
        self.cm_test(probs, y)

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
                 y.item(),
                 preds.item(),
                 logits.squeeze(), (y == preds).int().item()]
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



import pytorch_lightning as pl
import torch
import torch.nn as nn

class TwoOptimizerModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.automatic_optimization = False
        
        # Define your model components here
        self.dimReduction = nn.Linear(params.input_dim, params.hidden_dim)
        self.attention = nn.Linear(params.hidden_dim, 1)
        self.classifier = nn.Linear(params.hidden_dim, params.num_classes)
        self.attCls = nn.Linear(params.hidden_dim, params.num_classes)
        
        self.ce_cri = nn.CrossEntropyLoss()
        
        

    def forward(self, inputs):
        # Implement your forward pass here
        slide_sub_preds, gSlidePred, slide_pseudo_feat = self.process_inputs(inputs)
        return slide_sub_preds, gSlidePred, slide_pseudo_feat

    def process_inputs(self, inputs):
        # Implement the logic from your original code here
        pass

    def training_step(self, batch, batch_idx):
        # Get the optimizers
        opt0, opt1 = self.optimizers()
        
        # Enable gradient computation if it's not already enabled
        torch.set_grad_enabled(True)
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast():
            inputs, labels = batch
            slide_sub_preds, gSlidePred, _ = self(inputs)
            
            loss0 = self.ce_cri(slide_sub_preds, labels.repeat(self.params.numGroup)).mean()
            loss1 = self.ce_cri(gSlidePred, labels).mean()
        
        
