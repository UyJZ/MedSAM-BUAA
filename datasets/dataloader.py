import os
import cv2
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader


def get_bbox_from_mask(mask, target_size=1024):
    """Get bounding box [x_min, y_min, x_max, y_max] from binary mask.

    Args:
        mask (_type_): _description_
        target_size (int, optional): _description_. Defaults to 1024.

    Returns:
        np.ndarray: bounding box coordinates [x_min, y_min, x_max, y_max]
    """
    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) == 0:
        return np.array([0, 0, target_size, target_size])
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
    

class KVASIRDataset(Dataset):
    def __init__(self, root_dir, split='train', train_ratio=0.8, target_size=1024, seed=42):
        """
        KVASIR-SEG 数据集
        - KVASIR-SEG/
            - images/
                - img1.jpg
            - masks/
                - mask1.png

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

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        img_resized = self.preprocess(img, is_mask=False)
        mask_resized = self.preprocess(mask, is_mask=True)

        bbox = self.get_bbox_from_mask(mask_resized, self.target_size)

        return torch.tensor(img_resized).permute(2, 0, 1).float(), torch.tensor(mask_resized).unsqueeze(0).float(), torch.tensor(bbox).float(), name


class BraTSDataset(Dataset):
    """
    直接加载原始 BraTS NIfTI 数据集，
    自动切片为 2D + 自动生成 mask + 自动计算 bbox。
    """

    def __init__(self, root_dir, split="train", train_ratio=0.8, 
                 modality="flair", target_size=256, seed=42):
        """
        Args:
            root_dir: BraTS2021 数据集根目录
            split: train / val
            modality: 选择分割哪个模态 (flair / t1 / t1ce / t2)
            target_size: resize 尺寸
        """
        self.target_size = target_size
        self.modality = modality.lower()

        patient_dirs = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # train/val 划分
        np.random.seed(seed)
        np.random.shuffle(patient_dirs)

        split_idx = int(len(patient_dirs) * train_ratio)
        self.patient_dirs = (
            patient_dirs[:split_idx] if split == "train" else patient_dirs[split_idx:]
        )

        # 展开成 (patient, slice_index) 列表
        self.slice_list = []
        for p in self.patient_dirs:
            flair_path = os.path.join(p, f"{os.path.basename(p)}_{self.modality}.nii.gz")
            seg_path = os.path.join(p, f"{os.path.basename(p)}_seg.nii.gz")

            vol = nib.load(flair_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()

            # 每个 slice 都加入列表
            for i in range(vol.shape[2]):
                self.slice_list.append((p, i))

    def __len__(self):
        return len(self.slice_list)

    def preprocess(self, img, is_mask=False):
        if is_mask:
            return cv2.resize(img, (self.target_size, self.target_size),
                              interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_LINEAR)
            return (img - img.min()) / np.clip(img.max() - img.min(), 1e-8, None)

    def __getitem__(self, idx):
        patient_dir, slice_idx = self.slice_list[idx]
        pid = os.path.basename(patient_dir)

        # 加载模态
        img_nii = nib.load(os.path.join(patient_dir, f"{pid}_{self.modality}.nii.gz"))
        seg_nii = nib.load(os.path.join(patient_dir, f"{pid}_seg.nii.gz"))

        img_3d = img_nii.get_fdata()
        seg_3d = seg_nii.get_fdata()

        img = img_3d[:, :, slice_idx]
        seg = seg_3d[:, :, slice_idx]

        # BraTS 标签：0,1,2,4 → 二分类 mask
        seg = (seg > 0).astype(np.uint8)

        # 预处理
        img_resized = self.preprocess(img, is_mask=False)
        seg_resized = self.preprocess(seg, is_mask=True)

        # 计算 bbox
        bbox = get_bbox_from_mask(seg_resized, self.target_size)

        name = f"{pid}_slice_{slice_idx:03d}.png"

        return (
            torch.tensor(img_resized).unsqueeze(0).float(),  # [1,H,W]
            torch.tensor(seg_resized).unsqueeze(0).float(),  # [1,H,W]
            torch.tensor(bbox).float(),
            name
        )
        

# --- 测试 ---
if __name__ == "__main__":
    dataset_root = "./datasets/Kvasir-SEG"  # 你的数据集路径
    train_dataset = BraTSDataset("D:/Download/archive", split='train')
    val_dataset = BraTSDataset("D:/Download/archive", split='val')
    # train_dataset = KVASIRDataset(dataset_root, split='train')
    # val_dataset = KVASIRDataset(dataset_root, split='val')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    for batch in train_loader:
        image, mask, bbox, name = batch
        print(image.shape, mask.shape, bbox.shape)
        cv2.imshow("Image", image[0].permute(1, 2, 0).numpy())
        cv2.imshow("Mask", mask[0][0].numpy() * 255)
        cv2.waitKey(0)
        break
