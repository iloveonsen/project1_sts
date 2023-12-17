import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, outputs, targets):
        # Cosine similarity calculation
        cosine_similarity = F.cosine_similarity(outputs, targets, dim=-1)
        # Loss is defined as 1 - cosine_similarity to minimize the value during training
        loss = 1 - cosine_similarity
        # Return the mean loss over the batch
        return loss.mean()


class AdjustedCosineSimilarityLoss(nn.Module):
    def __init__(self, regression_loss_fn):
        super(AdjustedCosineSimilarityLoss, self).__init__()
        self.regression_loss_fn = regression_loss_fn

    def forward(self, outputs, targets):
        # Assuming outputs are normalized embeddings of the sentences
        # Calculate cosine similarity and scale it to be between 0 and 1
        cosine_similarity = F.cosine_similarity(outputs, targets, dim=-1)
        scaled_similarity = (cosine_similarity + 1.0) / 2.0

        # Adjust to the target range of 0 to 5
        adjusted_similarity = 5.0 * scaled_similarity

        # Calculate the loss as MSE with the actual target values
        loss = self.regression_loss_fn(adjusted_similarity, targets)

        return loss