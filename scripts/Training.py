import os
import torch
import wandb
import segmentation_models_pytorch as smp
import time
import pandas as pd

from datetime import datetime
from statistics import mean, stdev
from dotenv import dotenv_values
from torch import optim

from Utilities import (
    EarlyStopping,
    create_loader,
    create_dataset,
    wandb_setup,
    setup_env,
)
from config import Config, configure_params

CONFIG_DEFAULTS = None


def _init_train(model_name, config=None, run=None):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    early_stopping = EarlyStopping(
        model_dir=config.model_path, model_name=model_name, run=run, config=config
    )

    return early_stopping


def _train_epoch(
    config,
    model,
    dl_train,
    dl_valid,
    optimizer,
    criterion,
    metrics,
    scheduler=None,
    device="cuda",
):

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
    if scheduler:
        scheduler.step(valid_epoch_loss)

    return valid_epoch_loss, valid_epoch_iou


def _kfold_train(model_name, config, dataset, run=None, device="cuda"):
    best_losses = []
    best_ious = []

    for idx, (train_idx, valid_idx) in enumerate(dataset.folds):
        early_stopping = _init_train(model_name, config=config, run=run)
        model, kwargs = configure_params(config=config)

        criterion = kwargs["criterion"]
        optimizer = kwargs["optimizer"]
        metrics = [kwargs["metrics"]]
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", verbose=True, patience=3
        )

        if config.log:
            wandb.watch(model, log_graph=True)

        # Create loaders
        print(f"\nFold: {idx+1}\n-------")
        dl_train = create_loader(dataset, train_idx, batch_size=config.batch_size)
        dl_valid = create_loader(dataset, valid_idx, batch_size=config.batch_size)

        for epoch in range(1, config.epochs + 1):
            print(f"Epoch {epoch}\n")
            loss, iou = _train_epoch(
                config,
                model,
                dl_train,
                dl_valid,
                optimizer,
                criterion,
                metrics,
                lr_scheduler,
                device,
            )

            if early_stopping.checkpoint(
                model, epoch=epoch, loss=loss, iou=iou, optimizer=optimizer
            ):
                early_stopping.save_model()
                break

        best_losses.append(early_stopping.min_loss)
        best_ious.append(early_stopping.max_iou)

    return best_losses, best_ious


def _train(dataset, run=None, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.model_name

    if config.log:
        with run:
            best_losses, best_ious = _kfold_train(
                model_name=model_name,
                config=config,
                dataset=dataset,
                run=run,
                device=device,
            )
    else:
        best_losses, best_ious = _kfold_train(
            model_name=model_name,
            config=config,
            dataset=dataset,
            device=device,
        )

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


def train(defaults_path=""):

    start = datetime.now()
    print(f"\nConfiguration dataset complete. Training began at {start} ...\n")
    time.sleep(2)

    print(f"\nLoading configuration...")
    cfg = Config(
        defaults_path=defaults_path,
    )

    if cfg.log:
        setup_env(cfg.mode)
    print("Loading configuration complete.\n")

    # conf = dotenv_values("config/develop.env")

    # conf = (
    #     dotenv_values("config/develop.env")
    #     if config.mode == "develop"
    #     else dotenv_values("config/train.env")
    # )

    if cfg.sweep:
        sweep_cfg = cfg.sweep_cfg
        sweep_id = wandb.sweep(sweep_cfg, project="Sartorius-Kaggle-Competition")
        os.environ["WANDB_RUN_GROUP"] = sweep_id
        wandb.agent(sweep_id, sweep_train, count=cfg.count)

    else:
        if cfg.log:
            config, run = wandb_setup(cfg)
            ds_train = create_dataset(config=cfg)
            _train(config=config, run=run, dataset=ds_train)
        else:
            ds_train = create_dataset(config=cfg)
            _train(config=cfg, dataset=ds_train)

    end = datetime.now()
    delta = end - start
    print(f"\nTraining complete. Total training time {delta}.")
