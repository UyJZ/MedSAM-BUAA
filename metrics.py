from sklearn.metrics import jaccard_score
import torch
import numpy as np

def compute_metrics(preds, gt):
    """
    preds: Tensor, (B, 1, H, W) logits
    gt: Tensor, (B, 1, H, W)
    """
    preds_bin = (torch.sigmoid(preds) > 0.5).float()
    gt_bin = gt.float()

    batch_size = preds.shape[0]
    iou_list = []
    dice_list = []
    acc_list = []

    for b in range(batch_size):
        pred_flat = preds_bin[b].view(-1).cpu().numpy()
        gt_flat = gt_bin[b].view(-1).cpu().numpy()

        # IoU (Jaccard)
        iou = jaccard_score(gt_flat, pred_flat)
        iou_list.append(iou)

        # Dice
        dice = 2 * (pred_flat * gt_flat).sum() / max(pred_flat.sum() + gt_flat.sum(), 1e-8)
        dice_list.append(dice)

        # Accuracy
        acc = (pred_flat == gt_flat).mean()
        acc_list.append(acc)

    return np.mean(iou_list), np.mean(dice_list), np.mean(acc_list)
