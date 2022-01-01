import os
import torch
import wandb
import segmentation_models_pytorch as smp
import time
import pandas as pd

from datetime import datetime
from dotenv import dotenv_values
from statistics import mean, stdev

from Utilities import EarlyStopping, create_loader, CellDataset, reset_wandb_env
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


def _train(dataset, config=None, model_config=None, run=None):
    best_losses = []
    best_ious = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.model_name
    # scheduler = None if "scheduler" not in list(kwargs.keys()) else kwargs["scheduler"]

    for idx, (train_idx, valid_idx) in enumerate(dataset.folds):
        early_stopping = _init_train(
            model_name, config=config, checkpoint=config.checkpoint, run=run
        )
        model, kwargs = configure_params(config=config, model_cfg=model_config)

        criterion = kwargs["loss"]
        optimizer = kwargs["optimizer"]
        metrics = kwargs["metrics"]

        if config.log:
            wandb.watch(model, metrics[0], log_graph=True)

        # Create loaders
        print(f"\nFold: {idx+1}\n--------")
        dl_train = create_loader(dataset, train_idx, config=model_config)
        dl_valid = create_loader(dataset, valid_idx, config=model_config)

        for epoch in range(1, config.epochs + 1):
            print(f"Epoch {epoch}")
            print()

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
            print(
                f"\nTrain loss: {train_epoch_loss:.4f}\t Train iou: {train_epoch_iou:.4f}"
            )
            print(
                f"Validation loss: {valid_epoch_loss:.4f}\t Validation iou: {valid_epoch_iou:.4f}"
            )

            if early_stopping:
                breakpoint = early_stopping.checkpoint(
                    model,
                    epoch=epoch,
                    loss=valid_epoch_loss,
                    iou=valid_epoch_iou,
                    optimizer=optimizer,
                )

                if breakpoint:
                    best_losses.append(early_stopping.min_loss)
                    best_ious.append(early_stopping.max_iou)
                    early_stopping.save_model()
                    break

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


def setup(config=None):
    print("\nLoading training data...")
    df_train = pd.read_csv(config.train_csv)
    print("Loading training data complete.\n")
    print(df_train.info())

    print("\nConfiguring data...")
    ds_train = CellDataset(df_train, config=config)
    print("Configuring data complete.\n")

    return ds_train


def sweep_train(config=None):
    return
    # run = wandb.run
    # config = config if config else wandb.config
    # # ds_train, model, params = setup(config=config)
    # # _train(model=model, config=config, run=run, dataset=ds_train, kwargs=params)


def train(model_name, config=None):

    start = datetime.now()
    print(f"\nConfiguration setup complete. Training began at {start} ...\n")
    time.sleep(2)
    defaults_cfg = config.defaults_cfg
    model_cfg = config.model_cfg
    ds_train = setup(config=config)

    if config.log:
        reset_wandb_env()

        conf = (
            dotenv_values("config/develop.env")
            if config.mode == "develop"
            else dotenv_values("config/train.env")
        )

        github_sha = os.getenv("GITHUB_SHA")
        config.github_sha = github_sha[:5] if github_sha else None
        run_id = wandb.util.generate_id()

        os.environ["WANDB_API_KEY"] = conf["wandb_api_key"]
        os.environ["WANDB_ENTITY"] = conf["wandb_entity"]
        os.environ["WANDB_RESUME"] = conf["wandb_resume"]
        os.environ["WANDB_MODE"] = conf["wandb_mode"]
        os.environ["WANDB_JOB_TYPE"] = conf["wandb_job_type"]
        os.environ["WANDB_TAGS"] = conf["wandb_tags"]
        os.environ["WANDB_RUN_ID"] = run_id

        # if config.sweep:
        #     sweep_cfg = config.sweep_cfg
        #     sweep_id = wandb.sweep(sweep_cfg, project=conf["wandb_project"])

        run_name = "".join([config.model_name, f"-{run_id}"])
        run = wandb.init(
            project=conf["wandb_project"],
            entity=conf["wandb_entity"],
            config=defaults_cfg,
            reinit=True,
            name=run_name,
        )

        with run:
            config = wandb.config
            config["model_name"] = model_name
            if config.sweep:
                pass
                # wandb.agent(sweep_id, sweep_train, count=config.count)

            else:
                _train(
                    config=config,
                    model_config=model_cfg,
                    dataset=ds_train,
                    run=run,
                )
    # local implementation
    else:
        _train(
            config=config,
            model_config=model_cfg,
            dataset=ds_train,
        )

    end = datetime.now()
    delta = end - start
    print(f"\nTraining complete. Total training time {delta}.")
