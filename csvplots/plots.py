import json
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torchmetrics
from datetime import datetime
import math


# Calculate ensemble and plot from --inference --test
def ensemble_from_test_csv(target_path, target_column, pred_dir, pred_column, plot_dir="./plots", result_dir="./results", error_gap: float = 0.5):
    def get_bins(values):
        bin_min = math.floor(values.min().item()*10)/10
        bin_max = math.ceil(values.max().item()*10)/10
        bin_width = 0.1
        return np.arange(bin_min, bin_max + bin_width, bin_width)
            
    target_df = pd.read_csv(target_path)
    target_values = torch.tensor(target_df[target_column].values)
    pred_dir = Path(pred_dir)
    pred_file_paths = list(pred_dir.glob("*.csv"))

    ncols = 2
    nrows = len(pred_file_paths) + 1 # all predictions + ensemble
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), sharey='col')
    

    # Reading prediction files
    models_list = []
    model_predictions = []
    for i, pred_file_path in enumerate(pred_file_paths):
        model_name = pred_file_path.stem.split("_")[0]
        pred_df = pd.read_csv(pred_file_path)
        pred_values = torch.tensor(pred_df[pred_column].values)
        if target_values.shape != pred_values.shape:
            raise ValueError(f"The shape of target values {target_values.shape} and pred values {pred_values.shape} should be the same.")
        model_predictions.append(pred_values)
        models_list.append((model_name, pred_values))
    
    # ENSEMBLE
    # voting using softmax
    model_predictions = torch.stack(model_predictions, dim=0)
    model_scores = torch.nn.functional.softmax(model_predictions, dim=0)
    # print(f"Model predictions: {model_predictions.shape}, Model scores: {model_scores.shape}")
    assert model_predictions.shape == model_scores.shape
    # adopt score as weith
    model_results = model_predictions * model_scores # element-wise (weighted sum)
    model_results = model_results.sum(dim=0)
    if target_values.shape != model_results.shape:
        raise ValueError(f"The shape of target values {target_values.shape} and model results {model_results.shape} should be the same.")
    models_list.append(("Ensemble", model_results))

    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    target_df[pred_column] = model_results.numpy()
    target_df.to_csv(result_dir / f"ensemble_from_test_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    for i, (model_name, pred_values) in enumerate(models_list):
        metric = torchmetrics.functional.pearson_corrcoef(target_values, pred_values)
        error = torch.abs(target_values - pred_values)
        error_gap_mask = torch.where(error >= error_gap, 1, 0) # error gap 이상이면 1, 아니면 0
        error_count = error_gap_mask.sum().item()
        error_gap_color = ['red' if e.item() == 1 else 'blue' for e in error_gap_mask]

        # SCATTERPLOT [i][0]
        # Scatter plot for error_df
        sns.scatterplot(x=target_values, y=pred_values, color=error_gap_color, alpha=0.5, ax=axes[i,0])
        # Adding text labels for error_df
        for error_gap_idx in error_gap_mask.nonzero().flatten():
            axes[i,0].text(target_values[error_gap_idx], pred_values[error_gap_idx] + 0.1, error_gap_idx.item(), fontsize=8)
        # Reference line y=x
        sns.lineplot(x=[0, 5.2], y=[0, 5.2], color='black', ax=axes[i,0])
        sns.lineplot(x=[0, 4.5], y=[1, 5.5], color='black', linestyle='--', alpha=0.5, ax=axes[i,0])
        sns.lineplot(x=[1, 5.5], y=[0, 4.5], color='black', linestyle='--', alpha=0.5, ax=axes[i,0])
        # Set plot limits and title
        axes[i,0].set_xlim(-0.1, 5.5)
        axes[i,0].set_ylim(-0.1, 5.5)
        axes[i,0].set_title(f"Name: {model_name}\nMetric: {metric:.3f}\nError count: {error_count}")
        axes[i,0].set_xlabel("Origin target values")
        axes[i,0].set_ylabel("Model result")
        # Customizing plot appearance
        axes[i,0].spines['top'].set_visible(False)
        axes[i,0].spines['right'].set_visible(False)

        # HISTOGRAM [i][1]
        hist1_bins = get_bins(pred_values)
        hist1 = sns.histplot(pred_values, bins=hist1_bins, kde=True, color='blue', alpha=1, ax=axes[i,1])
        hist2_bins = get_bins(target_values)
        hist2 = sns.histplot(target_values, bins=hist2_bins, kde=True, color='red', alpha=.2, ax=axes[i,1])
        for p1, p2 in zip(hist1.patches, hist2.patches):
            if int(p1.get_height()) > 0:
                hist1.text(p1.get_x() + p1.get_width() / 2., p1.get_height(), int(p1.get_height()), fontsize=8, ha='center', va='bottom')
            if int(p2.get_height()) > 20:
                hist2.text(p2.get_x() + p2.get_width() / 2., p2.get_height(), int(p2.get_height()), fontsize=8, ha='center', va='bottom')
        axes[i,1].set_title(f"Name: {model_name}\nMetric: {metric:.3f}\nTarget bins: {len(hist2_bins)}, Pred bins: {len(hist1_bins)}")
        axes[i,1].set_xlim(-0.1, 5.5)
        axes[i,1].set_xlabel("Model result")
        axes[i,1].set_ylabel("Value counts")
        axes[i,1].spines['top'].set_visible(False)
        axes[i,1].spines['right'].set_visible(False)

    # Adjust layout to minimize spaces between plots
    fig.suptitle(f"Test model comparison\nError threshold: {error_gap}", fontsize=16)
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / f"ensemble_from_test_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()




# Calculate ensemble and plot from --inference
def ensemble_from_submission_csv(pred_dir, target_path="../data/sample_submission.csv", plot_dir="./plots", result_dir="./results"):
    def get_bins(values):
        bin_min = math.floor(values.min().item()*10)/10
        bin_max = math.ceil(values.max().item()*10)/10
        bin_width = 0.1
        return np.arange(bin_min, bin_max + bin_width, bin_width)
    
    target_df = pd.read_csv(target_path)
    example_target_values = torch.tensor(target_df["target"].values)

    pred_dir = Path(pred_dir)
    pred_file_paths = list(pred_dir.glob("*.csv"))

    ncols = 2
    nrows = int((len(pred_file_paths) + 2)/2) if len(pred_file_paths) % 2 == 0 else int((len(pred_file_paths) + 3)/2)
    # all predictions + ensemble + warpup
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), sharey=True)
    axes = axes.flatten()

    for ax in axes:
        ax.tick_params(axis='y', which='both', labelleft=True)

    # Reading prediction files
    models_list = []
    model_predictions = []
    for i, pred_file_path in enumerate(pred_file_paths):
        model_name = pred_file_path.stem.split("_")[0]
        pred_df = pd.read_csv(pred_file_path)
        pred_values = torch.tensor(pred_df["target"].values)
        if example_target_values.shape != pred_values.shape:
            raise ValueError(f"The shape of target values {example_target_values.shape} and pred values {pred_values.shape} should be the same.")
        model_predictions.append(pred_values)
        models_list.append((model_name, pred_values))
    
    # ENSEMBLE
    # voting using softmax
    model_predictions = torch.stack(model_predictions, dim=0)
    model_scores = torch.nn.functional.softmax(model_predictions, dim=0)
    # print(f"Model predictions: {model_predictions.shape}, Model scores: {model_scores.shape}")
    assert model_predictions.shape == model_scores.shape
    # adopt score as weith
    model_results = model_predictions * model_scores # element-wise (weighted sum)
    model_results = model_results.sum(dim=0)
    if example_target_values.shape != model_results.shape:
        raise ValueError(f"The shape of target values {example_target_values.shape} and model results {model_results.shape} should be the same.")
    models_list.append(("Ensemble", model_results))

    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    target_df["target"] = model_results.numpy()
    target_df.to_csv(result_dir / f"ensemble_from_submission_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    # print(f"len(models_list): {len(models_list)}, len(axes): {len(axes)}")

    last_idx = len(models_list)
    for i, (model_name, pred_values) in enumerate(models_list):
        # HISTOGRAM [i]
        bins = get_bins(pred_values)
        hist = sns.histplot(pred_values, bins=bins, kde=True, color='blue' if i != last_idx-1 else 'red', ax=axes[i])
        for p in hist.patches:
            if int(p.get_height()) > 0:
                hist.text(p.get_x() + p.get_width() / 2., p.get_height(), int(p.get_height()), fontsize=8, ha='center', va='bottom')
        axes[i].set_title(f"Name: {model_name}\nBins: {len(bins)}")
        axes[i].set_xlim(-0.1, 5.5)
        axes[i].set_xlabel("Model result")
        axes[i].set_ylabel("Value counts")
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        # WRAPUP HISTOGRAM [last_idx]
        alpha = .2
        if i == last_idx-1:
            alpha = 1 # clear for ensemble model result
        sns.histplot(pred_values, bins=get_bins(pred_values), kde=True, alpha=alpha, ax=axes[last_idx], color='blue' if i != last_idx-1 else 'red')
    
    axes[last_idx].set_title(f"Wrapup")
    axes[last_idx].set_xlim(-0.1, 5.5)
    axes[last_idx].set_xlabel("Model result")
    axes[last_idx].set_ylabel("Value counts")
    axes[last_idx].spines['top'].set_visible(False)
    axes[last_idx].spines['right'].set_visible(False)

    # Adjust layout to minimize spaces between plots
    fig.suptitle(f"Submission model comparison", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / f"ensemble_from_submission_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()