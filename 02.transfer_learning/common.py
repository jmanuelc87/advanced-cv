import os
import gc
import torch

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassF1Score,
)


def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        # Xavier uniform initialization for linear layers
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        # Kaiming uniform initialization (He initialization) for Conv2d layers
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.BatchNorm2d):
        # BatchNorm layers are often initialized differently
        torch.nn.init.normal_(m.weight, mean=1.0, std=0.02)
        torch.nn.init.zeros_(m.bias)


def train_model(
    model,
    config,
    optimizer: torch.optim.Optimizer,
    criterion,
    scheduler,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    device,
    train_metrics: list,
    valid_metrics: list,
    early_stopping=None,
    model_name=None,
    checkpoint_path=None,
    break_after_it=None,
):
    train_results = []
    valid_results = []

    N = config.EPOCHS * len(train_dataloader) + config.EPOCHS * len(valid_dataloader)
    progress = tqdm(total=N, desc="Training Progress", leave=True)

    scaler = torch.amp.GradScaler(device)  # type: ignore

    model = model.to(device)
    train_loss = float("inf")
    val_loss = float("inf")

    for epoch in range(config.EPOCHS):

        for metric in train_metrics:
            metric.reset()

        model.train()
        for i, batch in enumerate(train_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            if device == "cuda":
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            train_loss = loss.item() / inputs.size(0)

            pred = torch.argmax(outputs, dim=1)

            for metric in train_metrics:
                metric.update(pred.cpu(), labels.cpu())

            metrics = {
                f"Train{metric.__class__.__name__}": f"{metric.compute():.4f}"
                for metric in train_metrics
            }

            progress.set_postfix(
                {
                    "epoch": f"{epoch + 1}",
                    "loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    **metrics,
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                }
            )

            train_results.append(
                [epoch + ((i + 1) / len(train_dataloader)), train_loss]
                + [metric.compute() for metric in train_metrics]
            )

            progress.update(1)

            if break_after_it is not None and i > break_after_it:
                break

        model.eval()
        for i, batch in enumerate(valid_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss = loss.item()

            pred = torch.argmax(outputs, dim=1)

            for metric in valid_metrics:
                metric.update(pred.cpu(), labels.cpu())

            metrics = {
                f"Valid{metric.__class__.__name__}": f"{metric.compute():.4f}"
                for metric in valid_metrics
            }

            progress.set_postfix(
                {
                    "epoch": f"{epoch + 1}",
                    "loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    **metrics,
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                }
            )

            valid_results.append(
                [epoch + ((i + 1) / len(valid_dataloader)), val_loss]
                + [metric.compute() for metric in valid_metrics]
            )

            progress.update(1)

            if break_after_it is not None and i > break_after_it:
                break

        if (
            break_after_it is None
            and model_name is not None
            and checkpoint_path is not None
        ):
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.state_dict(),
                "loss": train_loss,
            }

            torch.save(
                checkpoint, os.path.join(checkpoint_path, f"epoch={epoch}_{model_name}")
            )

        if early_stopping is not None:
            early_stopping.check_early_stop(val_loss)

            if early_stopping.stop_training:
                print(f"Early stopping in epoch: {epoch}")
                break

    # Cleanup
    del model
    del optimizer
    del train_dataloader
    del valid_dataloader

    gc.collect()
    torch.cuda.empty_cache()

    return train_results, valid_results


def plot_trend(trends, labels, title, axis):
    from matplotlib.ticker import MaxNLocator

    plt.rcParams["figure.figsize"] = (14, 10)
    fig = plt.figure()
    fig.set_facecolor("white")

    plt.plot(
        trends[0][:][:, 0],
        trends[0][:][:, 1],
        color="tab:blue",
        label=labels[0],
    )

    plt.plot(
        trends[1][:][:, 0],
        trends[1][:][:, 1],
        color="tab:orange",
        label=labels[1],
    )

    plt.ylabel(axis)
    plt.legend(loc="upper center")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.title(title)
    plt.show()


def confusion_matrix(
    num_classes, model, dataloader, device, figsize=(10, 10), ticklabels=1
):
    confmat = MulticlassConfusionMatrix(num_classes=num_classes)

    model.eval()
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pred = torch.argmax(outputs, dim=1)
        confmat.update(pred.cpu(), labels.cpu())

    matrix = confmat.compute().numpy()

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        linewidths=3,
        cmap="coolwarm",
        ax=ax,
        cbar=False,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        square=True,
    )
    plt.show()


def evaluate_multi_classification_model(num_classes, model, dataloader, device):
    acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
    recall = MulticlassRecall(num_classes=num_classes, average="macro")
    precision = MulticlassPrecision(num_classes=num_classes, average="macro")
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    model.eval()
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        pred = torch.argmax(outputs, dim=1)

        acc.update(pred.cpu(), labels.cpu())
        recall.update(pred.cpu(), labels.cpu())
        precision.update(pred.cpu(), labels.cpu())
        f1.update(pred.cpu(), labels.cpu())

    return (
        acc.compute().item(),
        recall.compute().item(),
        precision.compute().item(),
        f1.compute().item(),
    )
