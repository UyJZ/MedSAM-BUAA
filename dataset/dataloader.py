import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib


class BraTSDataset(Dataset):
    def __init__(self, root, transform=None, mode="train"):
        """
        root: BraTS dataset directory
        transform: optional transforms
        mode: "train"/"val"/"test"
        """
        self.root = root
        self.transform = transform
        self.mode = mode

        # List all subjects
        self.samples = sorted([
            os.path.join(root, p) for p in os.listdir(root)
            if os.path.isdir(os.path.join(root, p))
        ])

    def __len__(self):
        return len(self.samples)

    def load_nii(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def __getitem__(self, idx):
        subject_dir = self.samples[idx]
        case_id = os.path.basename(subject_dir)

        # Load modals (shape: H x W x D)
        flair = self.load_nii(os.path.join(subject_dir, f"{case_id}_flair.nii.gz"))
        t1    = self.load_nii(os.path.join(subject_dir, f"{case_id}_t1.nii.gz"))
        t1ce  = self.load_nii(os.path.join(subject_dir, f"{case_id}_t1ce.nii.gz"))
        t2    = self.load_nii(os.path.join(subject_dir, f"{case_id}_t2.nii.gz"))

        # Load segmentation mask
        seg = self.load_nii(os.path.join(subject_dir, f"{case_id}_seg.nii.gz")).astype(np.uint8)

        # Stack into (C, H, W, D)
        img = np.stack([flair, t1, t1ce, t2], axis=0)

        # Optional intensity normalization (per volume)
        img = (img - img.mean()) / (img.std() + 1e-5)

        # Convert segmentation to WT, TC, ET if needed
        # BraTS labels:
        # 0: background
        # 1: edema
        # 2: non-enhancing tumor core
        # 4: enhancing tumor
        seg_ET = (seg == 4).astype(np.float32)
        seg_TC = ((seg == 4) | (seg == 1) | (seg == 2)).astype(np.float32)
        seg_WT = ((seg == 1) | (seg == 2) | (seg == 4)).astype(np.float32)

        label = np.stack([seg_WT, seg_TC, seg_ET], axis=0)

        # To tensor
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            img, label = self.transform(img, label)

        return img, label, case_id


if __name__ == "__main__":
    train_dataset = BraTSDataset(
        root="D:/Download/archive",
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for img, label, case_id in train_loader:
        print(img.shape)   # [1, 4, H, W, D]
        print(label.shape) # [1, 3, H, W, D]
        print(case_id)
        break
