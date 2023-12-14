from typing import Optional, Tuple, Union
import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from pytorch_lightning.callbacks import ModelCheckpoint

class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # plm: pretrained language model
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, **x):
        x = self.plm(**x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(**x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()

    # training_step 이전에 호출되는 함수입니다.
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    

# class CustomModelCheckpoint(ModelCheckpoint):
#     def _format_checkpoint_name(self, epoch, step, metrics, ver):
#         # Format the filename as you like
#         epoch_formatted = f"{epoch:03d}"
#         step_formatted = f"{step:05d}"
#         val_pearson_formatted = f"{metrics.get('val_pearson', 0.0):.3f}"

#         return f"_{epoch_formatted}_{step_formatted}_{val_pearson_formatted}_"