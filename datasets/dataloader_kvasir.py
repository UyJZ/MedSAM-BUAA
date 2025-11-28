import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KVASIRDataset(Dataset):
    def __init__(self, root_dir, split='train', train_ratio=0.8, target_size=1024, seed=42):
        """
        KVASIR-SEG 数据集

        Args:
            root_dir (str): 数据集根目录，里面必须有 'images' 和 'masks' 文件夹
            split (str): 'train' 或 'val'
            train_ratio (float): 训练集比例
            target_size (int): 图像和 mask 调整到的尺寸
            seed (int): 划分随机种子
        """
        self.target_size = target_size

        img_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # 获取所有图片文件
        all_images = sorted([
            f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        # 划分训练/验证集
        np.random.seed(seed)
        np.random.shuffle(all_images)
        split_idx = int(len(all_images) * train_ratio)
        self.images = all_images[:split_idx] if split == 'train' else all_images[split_idx:]

    def __len__(self):
        return len(self.images)

    def preprocess(self, img, is_mask=False):
        if is_mask:
            return cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            return (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)

    def get_bbox_from_mask(self, mask):
        y_idx, x_idx = np.where(mask > 0)
        if len(y_idx) == 0:
            return np.array([0, 0, self.target_size, self.target_size])
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        y_min, y_max = np.min(y_idx), np.max(y_idx)
        # 添加小扰动
        perturb = 5
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, perturb))
        x_max = min(W, x_max + np.random.randint(0, perturb))
        y_min = max(0, y_min - np.random.randint(0, perturb))
        y_max = min(H, y_max + np.random.randint(0, perturb))
        return np.array([x_min, y_min, x_max, y_max])

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        img_resized = self.preprocess(img, is_mask=False)
        mask_resized = self.preprocess(mask, is_mask=True)

        bbox = self.get_bbox_from_mask(mask_resized)

        return torch.tensor(img_resized).permute(2, 0, 1).float(), torch.tensor(mask_resized).unsqueeze(0).float(), torch.tensor(bbox).float(), name


# --- 测试 ---
if __name__ == "__main__":
    dataset_root = "Kvasir-SEG"  # 你的数据集路径
    train_dataset = KVASIRDataset(dataset_root, split='train')
    val_dataset = KVASIRDataset(dataset_root, split='val')

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    for batch in train_loader:
        print(batch['image'].shape, batch['mask'].shape, batch['bbox'].shape)
        break
