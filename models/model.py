from typing import Optional, Tuple, Union, List
import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Model(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # plm: pretrained language model
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        
        # special token의 embedding을 학습에 포함시킵니다.
        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5) # 야매로 5개 더 추가해줍니다.

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_fns = loss_fns
        

    def forward(self, **x):
        x = self.plm(**x)['logits']

        return x
    

    def custom_loss(self, outputs, targets):
        if len(self.loss_fns) == 1:
            loss = self.loss_fns[0](outputs, targets)
        elif len(self.loss_fns) < 1:
            raise ValueError("At least one loss function should be defined.")
        else:
            loss = 0
            for loss_fn in self.loss_fns:
                loss += loss_fn(outputs, targets)
            loss /= len(self.loss_fns)
        return loss
    

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("train_loss", loss)

        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        # return {"predictions": logits.squeeze(), "labels": y.squeeze(), "metric": torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())}

    # def on_test_epoch_end(self): # test_epoch_end(self, outputs) has beed deprecated and removed
    #     predictions = torch.cat([output["predictions"] for output in self.test_step_outputs])
    #     labels = torch.cat([output["labels"] for output in self.test_step_outputs])
    #     metric = torch.stack([output["metric"] for output in self.test_step_outputs]).mean()

    #     # self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(predictions, labels))
    #     # self.log("test_pearson", metric)
    #     # self.test_step_outputs.clear()
    #     return {"predictions": predictions, "labels": labels, "metric": metric}
    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()


    # training_step 이전에 호출되는 함수입니다.
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Define the LR scheduler
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True),
            'monitor': 'val_loss',  # Adjust this based on your validation metric
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]
    

# class CustomModelCheckpoint(ModelCheckpoint):
#     def _format_checkpoint_name(self, epoch, step, metrics, ver):
#         # Format the filename as you like
#         epoch_formatted = f"{epoch:03d}"
#         step_formatted = f"{step:05d}"
#         val_pearson_formatted = f"{metrics.get('val_pearson', 0.0):.3f}"

#         return f"_{epoch_formatted}_{step_formatted}_{val_pearson_formatted}_"