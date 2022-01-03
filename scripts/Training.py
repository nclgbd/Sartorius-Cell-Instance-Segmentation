import os
import torch
import wandb
import segmentation_models_pytorch as smp
import time
import pandas as pd

from datetime import datetime
from statistics import mean, stdev
from dotenv import dotenv_values

from Utilities import (
    EarlyStopping,
    create_loader,
    create_dataset,
    wandb_setup,
    setup_env,
)
from config import configure_params


def _init_train(model_name, config=None, checkpoint=True, run=None):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    early_stopping = None

    if checkpoint:
        if not config:
            raise ValueError("Cannot employ checkpointing without a configuration file")

        early_stopping = EarlyStopping(
            model_dir=config.model_path, model_name=model_name, run=run, config=config
        )

    return early_stopping


def _train_epoch(
    config,
    epoch,
    model,
    dl_train,
    dl_valid,
    optimizer,
    criterion,
    metrics,
    device="cuda",
):
    print(f"Epoch {epoch}\n")

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        device=device,
        verbose=True,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        device=device,
        verbose=True,
        metrics=metrics,
        loss=criterion,
    )

    train_logs = train_epoch.run(dl_train)
    train_logs_str = [f"{k} : {v:.4f}\t" for k, v in train_logs.items()]
    print("train:", "".join(train_logs_str))

    valid_logs = valid_epoch.run(dl_valid)
    valid_logs_str = [f"{k} : {v:.4f}\t" for k, v in valid_logs.items()]
    print("valid:", "".join(valid_logs_str))

    keys = list(train_logs.keys())
    if "mixed_loss" in keys:
        train_epoch_loss = train_logs["mixed_loss"]
        valid_epoch_loss = valid_logs["mixed_loss"]
    elif "dice_loss" in keys:
        train_epoch_loss = train_logs["dice_loss"]
        valid_epoch_loss = valid_logs["dice_loss"]

    train_epoch_iou = train_logs["iou_score"]
    valid_epoch_iou = valid_logs["iou_score"]

    if config.log:
        wandb.log({"train_logs": train_logs})
        wandb.log({"valid_logs": valid_logs})

    # Print epoch results
    print(f"\nTrain loss: {train_epoch_loss:.4f}\t Train iou: {train_epoch_iou:.4f}")
    print(
        f"Validation loss: {valid_epoch_loss:.4f}\t Validation iou: {valid_epoch_iou:.4f}"
    )


def _kfold_train(model_name, config, dataset, run=None, device="cuda"):
    for idx, (train_idx, valid_idx) in enumerate(dataset.folds):
        early_stopping = _init_train(
            model_name, config=config, checkpoint=config.checkpoint, run=run
        )
        model, kwargs = configure_params(config=config)

        criterion = kwargs["criterion"]
        optimizer = kwargs["optimizer"]
        metrics = [kwargs["metrics"]]

        if config.log:
            wandb.watch(model, log="all", log_graph=True)

        # Create loaders
        print(f"\nFold: {idx+1}\n--------")
        dl_train = create_loader(dataset, train_idx, batch_size=config.batch_size)
        dl_valid = create_loader(dataset, valid_idx, batch_size=config.batch_size)

        for epoch in range(1, config.epochs + 1):
            _train_epoch(
                early_stopping,
                config,
                epoch,
                model,
                dl_train,
                dl_valid,
                optimizer,
                criterion,
                metrics,
                device,
            )

            if early_stopping.check_patience():
                early_stopping.save_model()
                return early_stopping.best_loss, early_stopping.best_iou


def _train(dataset, run=None, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_losses = []
    best_ious = []
    model_name = config.model_name

    if config.log:
        with run:
            best_loss, best_iou = _kfold_train(
                model_name,
                config,
                dataset,
                run,
                device,
            )
    else:
        best_loss, best_iou = _kfold_train(
            model_name,
            config,
            dataset,
            device,
        )
    best_losses.append(best_loss)
    best_ious.append(best_iou)
    best_losses.append(best_loss)
    best_ious.append(best_iou)

    # Print all training and validation metrics
    if config.checkpoint:
        avg_iou = mean(best_ious)
        avg_iou_std = stdev(best_ious)

        avg_loss = mean(best_losses)
        avg_loss_std = stdev(best_losses)

        print(
            f"Average validation iou of all folds:\t{avg_iou:.4f} +/- {avg_iou_std:.4f}"
        )
        print(
            f"Average validation loss of all folds:\t{avg_loss:.4f} +/- {avg_loss_std:.4f}\n\n"
        )

        # Log the metrics if using wanb
        if config.log:
            avg_metrics = {}

            avg_metrics["avg_iou"] = avg_iou
            avg_metrics["avg_iou_std"] = avg_iou_std
            avg_metrics["avg_loss"] = avg_loss
            avg_metrics["avg_loss_std"] = avg_loss_std

            wandb.log({"avg_metrics": avg_metrics})


def sweep_train(config=None):
    config, run = wandb_setup(config)
    dataset = create_dataset(config=config)
    _train(dataset, run=run, config=config)


def train(config=None):

    start = datetime.now()
    print(f"\nConfiguration dataset complete. Training began at {start} ...\n")
    time.sleep(2)

    # conf = dotenv_values("config/develop.env")

    # conf = (
    #     dotenv_values("config/develop.env")
    #     if config.mode == "develop"
    #     else dotenv_values("config/train.env")
    # )

    setup_env(config.mode)

    if config.sweep:
        defaults_cfg = config.defaults_cfg
        sweep_cfg = config.sweep_cfg
        sweep_id = wandb.sweep(sweep_cfg, project="Sartorius-Kaggle-Competition")
        os.environ["WANDB_RUN_GROUP"] = sweep_id
        wandb.agent(sweep_id, sweep_train, count=config.count)

    else:
        ds_train = create_dataset(config=config)
        _train(config=config, dataset=ds_train)

    end = datetime.now()
    delta = end - start
    print(f"\nTraining complete. Total training time {delta}.")
