from typing import Optional, Tuple, Union, List
import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from torch.optim.lr_scheduler import StepLR, LambdaLR, SequentialLR, ReduceLROnPlateau

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

    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()
    
    # # training_step 이전에 호출되는 함수입니다.
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

    #     return optimizer

    # training_step 이전에 호출되는 함수입니다.
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Define the warm-up phase
        warmup_steps = 3
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / warmup_steps if epoch < warmup_steps else 1)

        # Define the StepLR scheduler
        step_size = 2  # Number of epochs between each step
        gamma = 0.9     # Multiplicative factor of learning rate decay
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Combine schedulers with SequentialLR
        schedulers = [warmup_scheduler, step_scheduler]
        milestones = [warmup_steps]  # The epochs at which to switch schedulers, here after warmup
        combined_scheduler = SequentialLR(optimizer, schedulers, milestones)

        return {"optimizer": optimizer, "lr_scheduler": combined_scheduler}
    



class SpecialTokenRegressionModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # plm: pretrained language model
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )

        self.regression_head = nn.Sequential(
            nn.Linear(3*self.plm.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(), # .5 default
            nn.Linear(128, 1),
        )
        
        # special token의 embedding을 학습에 포함시킵니다.
        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5) # 야매로 5개 더 추가해줍니다.

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_fns = loss_fns
        

    def forward(self, **x):
        # x = self.plm(**x)['logits']
        x = self.plm(**x)
        x = x.last_hidden_state[:, 0:3, :] # [batch_size, 3, hidden_size] = [batch_size, 3, 768]
        x = x.view(x.size(0), -1) # [batch_size, 3*hidden_size] = [batch_size, 3*768]
        x = self.regression_head(x)
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

    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()
    

    # # training_step 이전에 호출되는 함수입니다.
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

    #     return optimizer


    # training_step 이전에 호출되는 함수입니다.
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Define the warm-up phase
        warmup_steps = 3
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / warmup_steps if epoch < warmup_steps else 1)

        # Define the StepLR scheduler
        step_size = 2  # Number of epochs between each step
        gamma = 0.9     # Multiplicative factor of learning rate decay
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Combine schedulers with SequentialLR
        schedulers = [warmup_scheduler, step_scheduler]
        milestones = [warmup_steps]  # The epochs at which to switch schedulers, here after warmup
        combined_scheduler = SequentialLR(optimizer, schedulers, milestones)

        return {"optimizer": optimizer, "lr_scheduler": combined_scheduler}
    



class RDropRegressionModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.rdrop_alpha = 0.1

        # plm: pretrained language model
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )

        self.regression_head = nn.Sequential(
            nn.Linear(3*self.plm.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(), # .5 default
            nn.Linear(128, 1),
        )
        
        # special token의 embedding을 학습에 포함시킵니다.
        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5) # 야매로 5개 더 추가해줍니다.

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_fns = loss_fns
        

    def forward(self, **x):
        # x = self.plm(**x)['logits']
        x = self.plm(**x)
        x = x.last_hidden_state[:, 0:3, :] # [batch_size, 3, hidden_size] = [batch_size, 3, 768]
        x = x.view(x.size(0), -1) # [batch_size, 3*hidden_size] = [batch_size, 3*768]
        x = self.regression_head(x)
        return x
    

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits1 = self(**x)
        logits2 = self(**x)
        loss = self.custom_loss(logits1, y.float())

        rdrop_reg = self.rdrop_loss(logits1, logits2, alpha=self.rdrop_alpha, method="mse")

        total_loss = loss + rdrop_reg

        self.log("train_loss", total_loss)
        return total_loss
    

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
    

    def rdrop_loss(self, logits1, logits2, alpha=1.0, method="mse"):
        """
        Compute R-Drop regularization loss.
        
        Args:
            logits1 (torch.Tensor): Logits from the first forward pass.
            logits2 (torch.Tensor): Logits from the second forward pass.
            alpha (float): Weight of the R-Drop regularization term.
            method (str): The method for R-Drop loss ('mse' or 'l1').

        Returns:
            torch.Tensor: The R-Drop regularization loss.
        """
        if method == "mse":
            rdrop_reg = F.mse_loss(logits1, logits2)
        elif method == "l1":
            rdrop_reg = F.l1_loss(logits1, logits2)
        else:
            raise ValueError("Invalid method for R-drop loss. Use 'mse' or 'l1'.")
        
        return alpha * rdrop_reg
    

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

    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()
    

    # # training_step 이전에 호출되는 함수입니다.
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

    #     return optimizer


    # training_step 이전에 호출되는 함수입니다.
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Define the warm-up phase
        warmup_steps = 3
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / warmup_steps if epoch < warmup_steps else 1)

        # Define the StepLR scheduler
        step_size = 2  # Number of epochs between each step
        gamma = 0.9     # Multiplicative factor of learning rate decay
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Combine schedulers with SequentialLR
        schedulers = [warmup_scheduler, step_scheduler]
        milestones = [warmup_steps]  # The epochs at which to switch schedulers, here after warmup
        combined_scheduler = SequentialLR(optimizer, schedulers, milestones)

        return {"optimizer": optimizer, "lr_scheduler": combined_scheduler}




class RegressionModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # plm: pretrained language model
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name
        )

        self.regression_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.plm.config.hidden_size, 1),
        )
        
        # special token의 embedding을 학습에 포함시킵니다.
        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5) # 야매로 5개 더 추가해줍니다.

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_fns = loss_fns
        

    def forward(self, **x):
        # x = self.plm(**x)['logits']
        x = self.plm(**x)
        x = x.last_hidden_state[:, 0, :]
        x = self.regression_head(x)
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
    

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    #     warm_up_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / warm_up_epochs, 1.0))
    #     reduce_on_plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    #     return [{
    #         'optimizer': optimizer,
    #         'lr_scheduler': {
    #             'scheduler': warm_up_scheduler,
    #             'interval': 'step',
    #         }
    #     }, {
    #         'scheduler': reduce_on_plateau_scheduler,
    #         'interval': 'epoch',
    #         'frequency': 1,
    #         'monitor': 'val_loss',
    #     }]


    # training_step 이전에 호출되는 함수입니다.
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Define the LR scheduler
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True),
            'monitor': 'val_loss',  # Adjust this based on your validation metric
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [lr_scheduler]
    









