import os
from torch.utils.data import DataLoader
from datasets.dataloader import KVASIRDataset, GlaSDataset, BUSIDataset, ISICDataset, BraTSDataset
# from datasets.medsam_dataset import MedSAMDataset2D

# 数据集根目录映射
root_dir_dict = {
    'kvasir': 'Kvasir-SEG',
    # 'medsam': 'MedSAM',
    'glas': 'Warwick_QU_Dataset',
    'busi': 'Dataset_BUSI_with_GT',
    'isic': 'ISIC2018',
    'brats': 'BraTS2021' 
}

def build_dataloader(dataset_name, batch_size=2, num_workers=4, train_ratio=0.8, seed=42):
    """
    构建训练和测试 DataLoader，同时返回对应 Dataset

    Args:
        dataset_name (str): 数据集名称 ('kvasir' 或 'medsam')
        batch_size (int): 批大小
        num_workers (int): DataLoader 的 num_workers
        train_ratio (float): 训练集比例
        seed (int): 随机种子

    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in root_dir_dict:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    root_dir = os.path.join("./datasets", root_dir_dict[dataset_name])

    # 根据数据集名称实例化 Dataset
    if dataset_name == 'kvasir':
        train_dataset = KVASIRDataset(root_dir=root_dir, split='train', train_ratio=train_ratio, seed=seed)
        test_dataset  = KVASIRDataset(root_dir=root_dir, split='val',   train_ratio=train_ratio, seed=seed)
    # elif dataset_name == 'medsam':
    #     train_dataset = MedSAMDataset2D(root_dir=root_dir, split='train', train_ratio=train_ratio, seed=seed)
    #     test_dataset  = MedSAMDataset2D(root_dir=root_dir, split='val',   train_ratio=train_ratio, seed=seed)
    elif dataset_name == 'glas':
        train_dataset = GlaSDataset(root_dir=root_dir, split='train', train_ratio=train_ratio, seed=seed)
        test_dataset  = GlaSDataset(root_dir=root_dir, split='val',   train_ratio=train_ratio, seed=seed)
    elif dataset_name == 'busi':
        train_dataset = BUSIDataset(root_dir=root_dir, split='train', train_ratio=train_ratio, seed=seed)
        test_dataset  = BUSIDataset(root_dir=root_dir, split='val',   train_ratio=train_ratio, seed=seed)
    elif dataset_name == 'isic':
        train_dataset = ISICDataset(root_dir=root_dir, split='train', train_ratio=train_ratio, seed=seed)
        test_dataset  = ISICDataset(root_dir=root_dir, split='val', train_ratio=train_ratio, seed=seed)
    elif dataset_name == 'brats':
        train_dataset = BraTSDataset(root_dir=root_dir, split='train', train_ratio=train_ratio, seed=seed)
        test_dataset  = BraTSDataset(root_dir=root_dir, split='val',   train_ratio=train_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == "__main__":
    train_loader, test_loader, train_dataset, test_dataset = build_dataloader('kvasir', batch_size=4)

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    for batch in train_loader:
        print(batch['image'].shape, batch['mask'].shape, batch['bbox'].shape)
        break
