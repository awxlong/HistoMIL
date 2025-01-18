'''
Abstract class for a generic MIL model written in Pytorch Lightning.
'''
from abc import ABC, abstractmethod
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import wandb
from matplotlib import pyplot as plt
from HistoMIL.MODEL.Image.MIL.utils import  get_loss, get_optimizer, get_scheduler


class BaseMIL(pl.LightningModule, ABC):
    def __init__(self, model_paras):
        super().__init__()
        self.paras = model_paras           # Model-specific parameters
        self.model = self._create_model()  # Create the specific MIL model
        self.criterion = get_loss(model_paras.criterion) \
                         if model_paras.task == "binary" else get_loss(model_paras.criterion)
        self.save_hyperparameters()

        self.lr = model_paras.lr  # Learning rate
        self.wd = model_paras.wd  # Weight decay

        self._init_metrics()  # Initialize metrics

    @abstractmethod
    def _create_model(self):
        """Create and return the specific MIL model."""
        pass

    def _init_metrics(self):
        """Initialize metrics for training, validation, and testing."""
        task = self.paras.task
        num_classes = self.paras.num_classes

        # Training metrics
        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes)

        # Validation metrics
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.auroc_val = torchmetrics.AUROC(task=task, num_classes=num_classes)
        self.f1_val = torchmetrics.F1Score(task=task, num_classes=num_classes)
        self.precision_val = torchmetrics.Precision(task=task, num_classes=num_classes)
        self.recall_val = torchmetrics.Recall(task=task, num_classes=num_classes)
        self.specificity_val = torchmetrics.Specificity(task=task, num_classes=num_classes)
        self.cm_val = torchmetrics.ConfusionMatrix(task=task, num_classes=num_classes)

        # Testing metrics
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.auroc_test = torchmetrics.AUROC(task=task, num_classes=num_classes)
        self.f1_test = torchmetrics.F1Score(task=task, num_classes=num_classes)
        self.precision_test = torchmetrics.Precision(task=task, num_classes=num_classes)
        self.recall_test = torchmetrics.Recall(task=task, num_classes=num_classes)
        self.specificity_test = torchmetrics.Specificity(task=task, num_classes=num_classes)
        self.cm_test = torchmetrics.ConfusionMatrix(task=task, num_classes=num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = get_optimizer(
            name=self.paras.optimizer,
            model=self.model,
            lr=self.lr,
            wd=self.wd,
        )
        if self.paras.lr_scheduler:
            scheduler = get_scheduler(
                self.paras.lr_scheduler,
                optimizer,
                self.paras.lr_scheduler_config,
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch # x = WSI features, y = label
        logits = self.forward(x)
        if self.paras.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(0).float())
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1, keepdim=True)

        if self.paras.task == "binary":
            self.acc_train(preds, y.unsqueeze(1))
        else:
            probs = torch.softmax(logits, dim=1)
            self.acc_train(probs, y)
        self.log("acc_train", self.acc_train, prog_bar=True)
        self.log("loss_train", loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        logits = self.forward(x)
        if self.paras.task == "binary":
            y = y.unsqueeze(1)
            loss = self.criterion(logits, y.float())
            probs = torch.sigmoid(logits)
        else:
            loss = self.criterion(logits, y)
            probs = torch.softmax(logits, dim=1)

        self.acc_val(probs, y)
        self.auroc_val(probs, y)
        self.f1_val(probs, y)
        self.precision_val(probs, y)
        self.recall_val(probs, y)
        self.specificity_val(probs, y)
        self.cm_val(probs, y)

        self.log("loss_val", loss, prog_bar=True)
        self.log("acc_val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc_val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1_val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_val", self.precision_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall_val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("specificity_val", self.specificity_val, prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """Log validation confusion matrix at the end of the epoch."""
        if self.global_step != 0:
            cm = self.cm_val.compute()
            norm = cm.sum(axis=1, keepdims=True)
            normalized_cm = cm / norm
            plt.clf()
            cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
            wandb.log({"confusion_matrix_val": wandb.Image(cm)})
        self.cm_val.reset()

    def on_test_epoch_start(self):
        """Initialize outputs DataFrame at the start of the test epoch."""
        column_names = ['patient', 'ground_truth', 'prediction', 'probs', 'correct']
        self.outputs = pd.DataFrame(columns=column_names)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step."""
        x, y = batch
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
        self.log("precision_test", self.precision_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall_test", self.recall_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log("specificity_test", self.specificity_test, prog_bar=False, on_step=False, on_epoch=True)

        outputs = pd.DataFrame(
            data=[[y.item(), preds.item(), torch.sigmoid(logits.squeeze()).item(), (y == preds).int().item()]],
            columns=['ground_truth', 'prediction', 'probs', 'correct']
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)

    def on_test_epoch_end(self):
        """Log test confusion matrix at the end of the test epoch."""
        cm = self.cm_test.compute()
        norm = cm.sum(axis=1, keepdims=True)
        normalized_cm = cm / norm
        plt.clf()
        cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
        wandb.log({"confusion_matrix_test": wandb.Image(cm)})
        self.cm_test.reset()

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """Step the learning rate scheduler."""
        scheduler.step()

    def infer_step(self, batch):
        """Inference step."""
        self.eval()
        with torch.no_grad():
            x, y = batch
            logits, Y_prob, Y_hat, A_raw = self.model.infer(x)
        return logits, Y_prob, Y_hat, A_raw