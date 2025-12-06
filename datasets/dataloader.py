import os
import cv2
import numpy as np
import nibabel as nib
import random

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


class GlaSDataset(Dataset):
    # 数据集下载：https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation?resource=download
    def __init__(self, root_dir, split='train', train_ratio=0.8, target_size=1024, seed=42):
        """
        Args:
            root_dir (str): 包含所有图片和Mask的文件夹路径
            split (str): 'train' 或 'val'
            val_ratio (float): 验证集比例 (默认0.2，即20%做验证)
            seed (int): 随机种子，保证每次运行划分的结果一样
        """
        self.root_dir = root_dir
        self.target_size = target_size
        
        # 1. 获取所有文件名
        if not os.path.exists(root_dir):
            raise ValueError(f"路径不存在: {root_dir}")
            
        all_files = os.listdir(root_dir)

        # 2. 筛选原图 (排除带 _anno 的文件，且只看图片格式)
        # 逻辑：以 .bmp 结尾，且文件名里不包含 "_anno"
        self.images = [
            f for f in all_files 
            if f.lower().endswith(('.bmp', '.png', '.jpg')) and '_anno' not in f
        ]
        
        # 3. 排序并固定随机打乱 (保证训练集和验证集互斥且稳定)
        self.images.sort() 
        random.seed(seed)
        random.shuffle(self.images)

        # 4. 根据比例切分
        split_idx = int(len(self.images) * train_ratio)
        
        if split == 'train':
            self.images = self.images[:split_idx]  # 前 80%
        elif split == 'val':
            self.images = self.images[split_idx:]  # 后 20%
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.images)

    def preprocess(self, img, is_mask=False):
        if is_mask:
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            img = (img - img.min()) / np.clip(img.max() - img.min(), 1e-8, None)
        return img

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # --- 自动找对应的 Mask ---
        # 假设原图是 testA_1.bmp，Mask 必然是 testA_1_anno.bmp
        # 使用 rsplit 确保只替换扩展名部分
        name_part, ext_part = os.path.splitext(img_name)
        mask_name = f"{name_part}_anno{ext_part}"
        mask_path = os.path.join(self.root_dir, mask_name)

        # 1. 读取
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"找不到对应的Mask文件: {mask_path}")

        # 2. 二值化
        mask = (mask > 127).astype(np.uint8)

        # 3. 预处理
        img_resized = self.preprocess(img, is_mask=False)
        mask_resized = self.preprocess(mask, is_mask=True)
        mask_resized = mask_resized.astype(np.float32)

        # 4. Prompt
        bbox = get_bbox_from_mask(mask_resized, self.target_size)

        return (
            torch.tensor(img_resized).permute(2, 0, 1).float(),
            torch.tensor(mask_resized).unsqueeze(0).float(),
            torch.tensor(bbox).float(),
            img_name
        )


class BUSIDataset(Dataset):
    # 数据集下载：https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
    def __init__(self, root_dir, split='train', train_ratio=0.8, target_size=1024, seed=42):
        """
        Args:
            root_dir (str): 数据集根目录 (包含 benign/malignant/normal 子文件夹)
            split (str): 'train' 或 'val'
            val_ratio (float): 验证集比例
            target_size (int): 图像尺寸
            seed (int): 随机种子
        """
        self.root_dir = root_dir
        self.target_size = target_size

        if not os.path.exists(root_dir):
            raise ValueError(f"路径不存在: {root_dir}")

        # 1. 递归获取所有图片路径 (使用 os.walk 遍历子文件夹)
        self.images = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                # 筛选：是png图片 且 不是mask文件
                if f.lower().endswith('.png') and '_mask' not in f:
                    full_path = os.path.join(root, f)
                    self.images.append(full_path)

        if len(self.images) == 0:
            raise ValueError(f"在 {root_dir} 下未找到图片，请检查路径结构。")

        # 2. 排序并固定随机打乱
        self.images.sort()
        random.seed(seed)
        random.shuffle(self.images)

        # 3. 根据比例切分
        split_idx = int(len(self.images) * train_ratio)

        if split == 'train':
            self.images = self.images[:split_idx]
        elif split == 'val':
            self.images = self.images[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.images)

    def preprocess(self, img, is_mask=False):
        if is_mask:
            # Mask: 最近邻插值
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        else:
            # Image: 线性插值 + 归一化
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            img = (img - img.min()) / np.clip(img.max() - img.min(), 1e-8, None)
        return img

    def __getitem__(self, idx):
        # self.images[idx] 存储的是完整路径
        img_path = self.images[idx]

        # --- 自动找对应的 Mask ---
        # BUSI 规则: abc.png -> abc_mask.png
        # 因为 Mask 和原图在同一文件夹，直接替换路径字符串即可
        mask_path = img_path.replace(".png", "_mask.png")

        # 1. 读取
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # 简单容错：如果找不到 mask，返回全黑图（避免程序崩溃）
            # print(f"警告：找不到 Mask {mask_path}")
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # 2. 二值化
        mask = (mask > 127).astype(np.uint8)

        # 3. 预处理
        img_resized = self.preprocess(img, is_mask=False)
        mask_resized = self.preprocess(mask, is_mask=True)

        # 4. 生成 Prompt Bbox
        bbox = get_bbox_from_mask(mask_resized, self.target_size)

        # 5. 返回 Tensor (标准 PyTorch 写法)
        # Image: (3, H, W)
        # Mask:  (1, H, W)
        # Bbox:  (4)
        return (
            torch.tensor(img_resized).permute(2, 0, 1).float(),
            torch.tensor(mask_resized).unsqueeze(0).float(),
            torch.tensor(bbox).float(),
            os.path.basename(img_path)  # 只返回文件名
        )  

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

        bbox = get_bbox_from_mask(mask_resized, self.target_size)

        return torch.tensor(img_resized).permute(2, 0, 1).float(), torch.tensor(mask_resized).unsqueeze(0).float(), torch.tensor(bbox).float(), name


class BraTSDataset(Dataset):
    """
    直接加载原始 BraTS NIfTI 数据集，
    自动切片为 2D + 自动生成 mask + 自动计算 bbox。
    """

    def __init__(self, root_dir, split="train", train_ratio=0.8, 
                 modality="flair", target_size=1024, seed=42):
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
        

class ISICDataset(Dataset):
    """
    使用 ISIC 2018 官方训练集（唯一有 mask 的部分），
    并在本地自行划分 train/val。
    """

    def __init__(self, root_dir, split='train', target_size=1024,
                 train_ratio=0.8, seed=42):
        self.target_size = target_size

        img_dir = os.path.join(root_dir, "ISIC2018_Task1-2_Training_Input")
        mask_dir = os.path.join(root_dir, "ISIC2018_Task1_Training_GroundTruth")

        assert os.path.exists(img_dir), f"{img_dir} 不存在"
        assert os.path.exists(mask_dir), f"{mask_dir} 不存在"

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        all_images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])
        
        np.random.seed(seed)
        np.random.shuffle(all_images)
        split_idx = int(len(all_images) * train_ratio)

        if split == "train":
            self.images = all_images[:split_idx]
        elif split == "val":
            self.images = all_images[split_idx:]
        else:
            raise ValueError("split 必须是 train 或 val")

    def __len__(self):
        return len(self.images)

    def preprocess(self, img, is_mask=False):
        if is_mask:
            return cv2.resize(img, (self.target_size, self.target_size),
                              interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_LINEAR)
            return (img - img.min()) / np.clip(img.max() - img.min(), 1e-8, None)

    def __getitem__(self, idx):
        name = self.images[idx]

        img_path = os.path.join(self.img_dir, name)

        # mask 文件名格式：xxx_segmentation.png
        mask_name = name.replace(".png", "_segmentation.png") \
                        .replace(".jpg", "_segmentation.png") \
                        .replace(".JPG", "_segmentation.png") \


        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image & mask
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        print(img_path, mask_path)
        mask = (mask > 127).astype(np.uint8)

        img_resized = self.preprocess(img, is_mask=False)
        mask_resized = self.preprocess(mask, is_mask=True)

        bbox = get_bbox_from_mask(mask_resized, self.target_size)

        return (
            torch.tensor(img_resized).permute(2, 0, 1).float(),
            torch.tensor(mask_resized).unsqueeze(0).float(),
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
