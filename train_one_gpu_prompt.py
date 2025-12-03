# -*- coding: utf-8 -*-
"""
train the image encoder (with prompt tuning) and mask decoder
freeze original image encoder parameters and prompt encoder
"""

# %% setup environment
from metrics import compute_metrics
from build_dataloader import build_dataloader
import glob
import shutil
from datetime import datetime
import random
import argparse
import torch.nn.functional as F
from segment_anything_med import sam_model_registry
import monai
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from skimage import transform
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join

# set seeds
torch.manual_seed(42)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue",
                      facecolor=(0, 0, 0, 0), lw=2)
    )


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/npy/CT_Abd",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument(
    "-dataset_name", type=str, default="kvasir", help="dataset name"
)
parser.add_argument("-task_name", type=str, default="MedSAM-Prompt")
parser.add_argument("-model_type", type=str, default="vit_prompt")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name, args.model_type + "-" + run_id)
device = torch.device(args.device)
# %% set up model


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))

    # build dataloaders
    train_loader, val_loader, train_dataset, val_dataset = build_dataloader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.9,
        seed=42,
    )

    # load SAM model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    # Unfreeze mask_decoder (because sam_prompt.py froze everything initially)
    for param in medsam_model.mask_decoder.parameters():
        param.requires_grad = True

    medsam_model.train()

    # optimizer
    # Only train parameters that require grad (prompt embeddings and mask decoder)
    trainable_params = [p for p in medsam_model.parameters() if p.requires_grad]

    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # losses
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_loss = float("inf")
    losses = []

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0
        medsam_model.train()
        for step, (images, gt2D, boxes, _) in enumerate(tqdm(train_loader)):
            images, gt2D = images.to(device), gt2D.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                preds = medsam_model(images, boxes.to('cpu').numpy())
                loss = seg_loss(preds, gt2D) + ce_loss(preds, gt2D.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)

        # validation
        medsam_model.eval()
        val_loss = 0
        all_iou, all_dice, all_acc = [], [], []

        with torch.no_grad():
            for images, gt2D, boxes, _ in val_loader:
                images, gt2D = images.to(device), gt2D.to(device)
                preds = medsam_model(images, boxes.to('cpu').numpy())
                loss = seg_loss(preds, gt2D) + ce_loss(preds, gt2D.float())
                val_loss += loss.item()

                # compute metrics
                iou, dice, acc = compute_metrics(preds, gt2D)
                all_iou.append(iou)
                all_dice.append(dice)
                all_acc.append(acc)

        val_loss /= len(val_loader)
        mean_iou = np.mean(all_iou)
        mean_dice = np.mean(all_dice)
        mean_acc = np.mean(all_acc)

        print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {mean_iou:.4f} | Val Dice: {mean_dice:.4f} | Val Acc: {mean_acc:.4f}")

        # save checkpoints
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(
            model_save_path, "medsam_model_latest.pth"))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, join(
                model_save_path, "medsam_model_best.pth"))

        # plot loss
        plt.figure()
        plt.plot(losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, f"{args.task_name}_train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
