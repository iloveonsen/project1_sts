import argparse
import random
from utils.utils import read_json 
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from itertools import product
import os

import pandas as pd

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


import wandb
from pytorch_lightning.loggers import WandbLogger

from utils.utils import *
from models.model import *
# from models.callbacks import *
from data.data_module import *
from models.loss import *


os.environ["TZ"] = "Asia/Seoul"

def main(config: Dict):
    # seed 고정
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    random.seed(config["seed"])

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', default=config["inference"], action="store_true")
    parser.add_argument('--best', default=config["best"], action="store_true") # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    parser.add_argument('--test', default=config["test"], action="store_true")
    parser.add_argument('--resume', default=config["resume"], action="store_true")
    parser.add_argument('--shuffle', default=config["shuffle"], action="store_true")
    parser.add_argument('--wandb_project_name', default=config["wandb_project_name"], type=str)
    parser.add_argument('--wandb_username', default=config["wandb_username"], type=str)
    parser.add_argument('--model_name', default=config["model_name"], type=str)
    parser.add_argument('--model_detail', default=config["model_detail"], type=str) # 돌리는 모델의 detail 추가
    parser.add_argument('--batch_size', default=config["batch_size"], type=int)
    parser.add_argument('--max_epoch', default=config["max_epoch"], type=int)
    parser.add_argument('--learning_rate', default=config["learning_rate"], type=float)
    parser.add_argument('--kfold', default=config["kfold"], type=int)
    parser.add_argument('--data_dir', default=config["data_dir"])
    parser.add_argument('--model_dir', default=config["model_dir"])
    parser.add_argument('--test_output_dir', default=config["test_output_dir"])
    parser.add_argument('--output_dir', default=config["output_dir"])
    parser.add_argument('--train_path', default=config["train_path"])
    parser.add_argument('--dev_path', default=config["dev_path"])
    parser.add_argument('--test_path', default=config["test_path"])
    parser.add_argument('--predict_path', default=config["predict_path"])

    args = parser.parse_args()
    # print(f"inference: {args.inference}, best: {args.best}, resume: {args.resume}, shuffle: {args.shuffle}")

    train_path = Path(args.data_dir) / args.train_path
    dev_path = Path(args.data_dir) / args.dev_path
    test_path = Path(args.data_dir) / args.test_path
    predict_path = Path(args.data_dir) / args.predict_path
    

    if not args.inference:
        print("Start training...")

        if len(args.model_name) != len(args.model_detail):
            raise ValueError("The number of model_name and model_detail should be the same.")
        
        for model_name, model_detail in zip(args.model_name, args.model_detail):
            print(f"Current model_name: {model_name}, model_detail: {model_detail}")
    
            grids = list(product(args.batch_size, args.max_epoch, args.learning_rate))
            print(f"Total {len(grids)} combinations has been detected...")

            for i, combination in enumerate(grids, start=1):
                batch_size, max_epoch, learning_rate = combination
                
                print(f"#{i}" + "=" * 80)
                print(f"model_name: {model_name}, model_detail: {model_detail}\nbatch_size: {batch_size}\nmax_epoch: {max_epoch}\nlearning_rate: {learning_rate}\n")
                latest_version, _ = get_latest_version(args.model_dir, model_name)

                # model_dir / model_provider / model_name + model_version + batch_size + max_epoch + learning_rate + current_epoch + current_step + eval_metric + YYYYMMDD + HHMMSS + .ckpt
                # ./saves/klue/roberta-small_v03_16_1_1e-05_000_00583_0.862_20231214_221830.ckpt 같은 형태로 저장됩니다
                save_name = "-".join(model_name.split("/")[1].split()) + "_" + "-".join(model_detail.split()) + prefix_zero(latest_version + 1, 2) + f"_{batch_size}_{max_epoch}_{learning_rate}"
                print(f"save_name: {save_name}")
                wandb_logger = WandbLogger(project=args.wandb_project_name, entity=args.wandb_username)

                # model을 생성합니다.
                early_stop_custom_callback = EarlyStopping("val_pearson", patience=6, verbose=True, mode="max")

                model_provider = model_name.split("/")[0] # "klue"/roberta-large
                dirpath = Path(args.model_dir) / model_provider
                dirpath.mkdir(parents=True, exist_ok=True)

                checkpoint_callback = ModelCheckpoint(
                    monitor="val_pearson",
                    save_top_k=1,
                    dirpath=dirpath,
                    filename=save_name + "_{epoch:03d}_{step:05d}_" + "{val_pearson:0.3f}" + "_" + datetime.today().strftime("%Y%m%d_%H%M%S"),
                    auto_insert_metric_name=False,
                    save_weights_only=False,
                    verbose=True,
                    mode="max"
                )

                # Configure loss function
                loss_fns = [nn.SmoothL1Loss()]
                if len(loss_fns) > 1:
                    print(f"Mutiple loss functions are detected. Loss functions will be summed up.")

                model = RegressionModel(model_name, learning_rate, loss_fns)
                # model.load_state_dict(checkpoint['state_dict'])

                num_folds = args.kfold
                split_seed = config["seed"]
                if num_folds > 1:
                    print(f"KFold dataloader will be used. nums_folds: {num_folds}, split_seed: {split_seed}")
                    results = []

                    # nums_folds는 fold의 개수, k는 k번째 fold datamodule
                    for k in range(num_folds):
                        print(f"Current fold: {k}th fold" + "=" * 80)
                        kfdataloader = KFoldDataloader(model_name, batch_size, args.shuffle, train_path, dev_path, test_path, predict_path,
                                                        k=k, split_seed=split_seed, num_splits=num_folds)
                        kfdataloader.prepare_data()
                        kfdataloader.setup()

                        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epoch//num_folds, 
                                             callbacks=[checkpoint_callback,early_stop_custom_callback],
                                             log_every_n_steps=1,logger=wandb_logger)

                        trainer.fit(model=model, datamodule=kfdataloader)
                        score = trainer.test(model=model, datamodule=kfdataloader)

                        results.extend(score)

                    result = [x['test_pearson'] for x in results]
                    score = sum(result) / num_folds
                    print(f"K fold Test score: {score}" + "=" * 80)

                else:
                    dataloader = Dataloader(model_name, batch_size, args.shuffle, train_path, dev_path, test_path, predict_path)

                    # special token의 embedding을 학습에 포함시킵니다.
                    # print(f"Dataset tokenizer vocab size: {len(dataloader.tokenizer)}, Model tokenizer vocab size: {model.plm.get_input_embeddings().num_embeddings}")
                    # model.plm.resize_token_embeddings(len(dataloader.tokenizer)) 
                    # print(f"Aftrer updating Model tokenizer vocab size: {model.plm.get_input_embeddings().num_embeddings}")

                    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
                    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epoch, 
                                         callbacks=[checkpoint_callback,early_stop_custom_callback],
                                         log_every_n_steps=1,logger=wandb_logger)

                    # Train part
                    trainer.fit(model=model, datamodule=dataloader)
                    trainer.test(model=model, datamodule=dataloader)

                # 학습이 완료된 모델을 저장합니다.
                torch.save(model, dirpath / f"{save_name}.pt")

    else:
        print("Start inference...")
        if len(args.model_name) > 1:
            print("Multiple models are detected. Only the first model will be used for inference.")
        model_name = args.model_name[0]

        if args.best:
            print("Loading the best performance model...")
            select_version, select_version_perf, select_version_path = get_version(args.model_dir, model_name, best=True)
        else:
            print("Loading latest trained model...")
            select_version, select_version_perf, select_version_path = get_version(args.model_dir, model_name)
        batch_size = int(select_version_path.stem.split("_")[-8])
        
        print(f"#inference" + "=" * 80)
        print(f"model_name: {model_name}\nversion: v{select_version}\nval_perf: {select_version_perf}\nbatch_size: {batch_size}\n")

        trainer = pl.Trainer(accelerator="gpu", 
                             devices=1, max_epochs=1)
        model = RegressionModel.load_from_checkpoint(select_version_path)

        output_dir = Path(args.output_dir) if not args.test else Path(args.test_output_dir)
        model_provider = model_name.split("/")[0] # "klue"/roberta-large
        output_path = output_dir / model_provider
        output_path.mkdir(parents=True, exist_ok=True)

        if args.test:
            print(f"\nInference on test dataset {test_path}...")
            dataloader = Dataloader(model_name, batch_size, False, train_path, dev_path, test_path, test_path) # prediction with dev.csv
            test_predictions = trainer.predict(model=model, datamodule=dataloader)
            test_predictions = list(round(val.item(), 1) for val in torch.cat(test_predictions))
            
            # Aggregate batch outputs into one
            output = pd.read_csv(test_path)
            output["predict"] = test_predictions
            output = output.drop(columns=["binary-label"])
            metric = torchmetrics.functional.pearson_corrcoef(torch.tensor(output["predict"]), torch.tensor(output["label"]))
            output_file_name = '_'.join(select_version_path.stem.split("_")[:-3]) + f"_{metric:.3f}_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"  
        else:
            print(f"\nInference for submission {predict_path}...")
            dataloader = Dataloader(model_name, batch_size, False, train_path, dev_path, test_path, predict_path)
            predictions = trainer.predict(model=model, datamodule=dataloader)
            predictions = list(round(val.item(), 1) for val in torch.cat(predictions)) # (# batches, batch_size * 1) -> (# batches * batch_size * 1)
            
            output = pd.read_csv("./data/sample_submission.csv")
            output["target"] = predictions
            output_file_name = '_'.join(select_version_path.stem.split("_")[:-2]) + f"_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv" # add prediction time
        output.to_csv(output_path / output_file_name, index=False)


if __name__ == '__main__':
    config = read_json('./config/config.json')
    main(config=config)
