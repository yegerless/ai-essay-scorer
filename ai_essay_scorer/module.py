# module.py
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics.classification import AUROC, Accuracy, F1Score
from transformers import AutoModelForSequenceClassification

# from lightning.pytorch.loggers import MLFlowLogger


class LightningTextClassifier(pl.LightningModule):
    def __init__(
        self, hf_model: str, lr: float, weight_decay: float, freeze_n_layers, num_classes: int
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model, num_labels=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        # Freeze bottom-N-layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # self.example_input_array = torch.Tensor(32, 1, 28, 28)

        # accuracy
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        # roc_auc
        self.train_rocauc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.val_rocauc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.test_rocauc = AUROC(task="multiclass", num_classes=self.num_classes)

        # f1-score
        self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=self.num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        return self.model(input_ids, attention_mask).logits

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        y = batch["labels"].long()
        logits = self(**batch)
        loss = self.criterion(logits, y)
        pred_probs = nn.functional.softmax(logits, dim=-1)
        pred_classes = torch.argmax(pred_probs, dim=-1)

        accuracy = self.train_accuracy(pred_classes, y)
        roc_auc = self.train_rocauc(pred_probs, y)
        f1 = self.train_f1(pred_classes, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_roc_auc": roc_auc,
                "train_f1_score": f1,
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        y = batch["labels"].long()
        logits = self.forward(**batch)
        val_loss = self.criterion(logits, y)
        pred_probs = nn.functional.softmax(logits, dim=-1)
        pred_classes = torch.argmax(pred_probs, dim=-1)

        accuracy = self.val_accuracy(pred_classes, y)
        roc_auc = self.val_rocauc(pred_probs, y)
        f1 = self.val_f1(pred_classes, y)
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_accuracy": accuracy,
                "val_roc_auc": roc_auc,
                "val_f1_score": f1,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        y = batch["labels"].long()
        logits = self.forward(**batch)
        test_loss = self.criterion(logits, y)
        pred_probs = nn.functional.softmax(logits, dim=-1)
        pred_classes = torch.argmax(pred_probs, dim=-1)

        accuracy = self.test_accuracy(pred_classes, y)
        roc_auc = self.test_rocauc(pred_probs, y)
        f1 = self.val_f1(pred_classes, y)
        self.log_dict(
            {
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "test_roc_auc": roc_auc,
                "test_f1_score": f1,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay, fused=True
        )
        return optimizer
