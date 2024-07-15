import os
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.transform import *
# from src.models import BasicConvClassifier
from src.conv_model import AdvancedConvClassifier
from src.loss import FocalCrossEntropyLoss
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_transform = transforms.Compose([
        NormalizeTransform(),
        AddNoiseTransform(noise_level=0.01),
        RandomCropTransform(crop_size=200),  # Adjust crop size as needed
        RandomTimeWarpTransform(),
        RandomScalingTransform(),
        RandomErasingTransform(p=0.5)
    ])

    test_transform = transforms.Compose([
        NormalizeTransform()
    ])

    # loader
    # wave data X: (65728, 271, 281) ... (batch_size, num_channels, seq_len)
    # class label Y: (65728,) ... (batch_size,) 1854 classes 0-1853
    # subject index: (65728,) ... (batch_size,) 4 subjects 0-3
    # transformer
    # NOTE: va and test only need normalization
    train_set = ThingsMEGDataset("train", args.data_dir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    # model = BasicConvClassifier(
    #     train_set.num_classes, train_set.seq_len, train_set.num_channels
    # ).to(args.device)
    model = AdvancedConvClassifier(
        num_classes=train_set.num_classes,
        num_subjects=4,
        in_channels=train_set.num_channels,
        seq_len=train_set.seq_len,
    ).to(args.device)

    # New Loss Function
    criterion = FocalCrossEntropyLoss()

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=model.l2_reg)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X, subject_idxs.to(args.device))

            # loss = F.cross_entropy(y_pred, y)
            loss = criterion(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = model(X, subject_idxs.to(args.device))

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(
            f"Epoch {epoch + 1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss),
                 "val_acc": np.mean(val_acc)})

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
