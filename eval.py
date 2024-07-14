import os
import numpy as np
import torch
from torchvision import transforms
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.transform import NormalizeTransform
from src.conv_model import AdvancedConvClassifier
from src.utils import set_seed


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)

    # ------------------
    #    Dataloader
    # ------------------
    test_transform = transforms.Compose([
        NormalizeTransform()
    ])
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
        num_classes=test_set.num_classes,
        num_subjects=4,
        in_channels=test_set.num_channels,
        seq_len=test_set.seq_len,
    ).to(args.device)

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()
