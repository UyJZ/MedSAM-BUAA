import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


class MedSAMDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir (str): 图像文件夹路径
            mask_dir (str): 标注(Mask)文件夹路径
            transform (callable, optional): 数据增强/预处理
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 获取所有文件名 (假设图片和Mask文件名一致，即使后缀不同也可以处理，这里假设一致)
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        # MedSAM 标准输入尺寸通常是 1024x1024
        self.target_size = 1024

    def __len__(self):
        return len(self.images)

    def preprocess(self, img, is_mask=False):
        """
        简单的预处理：调整大小到 1024x1024
        """
        if is_mask:
            # Mask 使用最近邻插值，保持 0/1 整数
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        else:
            # 图像使用线性插值
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            # 归一化 (MedSAM官方代码通常在模型内部或此处做归一化，这里先转为 0-1 范围)
            img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)
        return img

    def get_bbox_from_mask(self, mask):
        """
        从 Mask 中提取 Bounding Box [x_min, y_min, x_max, y_max]
        作为 MedSAM 的 Prompt
        """
        y_indices, x_indices = np.where(mask > 0)

        # 如果 Mask 是全黑的（没有目标），返回一个空框或全图框
        if len(y_indices) == 0:
            return np.array([0, 0, self.target_size, self.target_size])

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # 增加一点随机扰动（噪声），模拟人工提示的不完美，有助于训练鲁棒性
        H, W = mask.shape
        add_noise = True
        if add_noise:
            perturb = 5  # 像素扰动
            x_min = max(0, x_min - np.random.randint(0, perturb))
            x_max = min(W, x_max + np.random.randint(0, perturb))
            y_min = max(0, y_min - np.random.randint(0, perturb))
            y_max = min(H, y_max + np.random.randint(0, perturb))

        return np.array([x_min, y_min, x_max, y_max])

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)  # 假设文件名相同

        # 1. 读取图像 (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 读取Mask (Grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 简单二值化，确保 mask 只有 0 和 1 (有的数据集是0-255)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # 3. 预处理 (Resize)
        image_resized = self.preprocess(image, is_mask=False)
        mask_resized = self.preprocess(mask, is_mask=True)

        # 4. 生成 Prompt (Bounding Box)
        # 格式: [x1, y1, x2, y2]
        bbox = self.get_bbox_from_mask(mask_resized)

        # 5. 转换为 Tensor
        # Image: (C, H, W) -> (3, 1024, 1024)
        image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float()
        # Mask: (1, H, W) -> (1, 1024, 1024)
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
        # Bbox: (4)
        bbox_tensor = torch.tensor(bbox).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
            "name": img_name
        }


# --- 测试代码 ---
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 使用 os.path.join 拼接路径
    # 假设 'Kvasir-SEG' 文件夹就在这个代码文件的旁边
    IMG_DIR = os.path.join(current_script_dir, "Kvasir-SEG", "images")
    MASK_DIR = os.path.join(current_script_dir, "Kvasir-SEG", "masks")

    # 检查路径是否存在，避免报错
    if os.path.exists(IMG_DIR) and os.path.exists(MASK_DIR):
        dataset = MedSAMDataset(img_dir=IMG_DIR, mask_dir=MASK_DIR)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        print(f"Dataset Size: {len(dataset)}")

        # 取一个 batch 查看数据
        for batch in dataloader:
            imgs = batch['image']
            masks = batch['mask']
            bboxes = batch['bbox']

            print(f"Image Batch Shape: {imgs.shape}")  # 应为 [B, 3, 1024, 1024]
            print(f"Mask Batch Shape: {masks.shape}")  # 应为 [B, 1, 1024, 1024]
            print(f"Bbox Batch Shape: {bboxes.shape}")  # 应为 [B, 4]
            print("Bbox Example:", bboxes[0])

            # 可视化第一张图验证对齐情况
            plt.figure(figsize=(10, 5))

            # 显示原图
            plt.subplot(1, 2, 1)
            img_np = imgs[0].permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            # 画出 Bbox
            x1, y1, x2, y2 = bboxes[0].numpy()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.title("Image with Bbox Prompt")

            # 显示Mask
            plt.subplot(1, 2, 2)
            plt.imshow(masks[0].squeeze().numpy(), cmap='gray')
            plt.title("Ground Truth Mask")

            plt.show()
            break  # 只看第一个batch
    else:
        print("请先创建数据集文件夹并放入图片进行测试！")