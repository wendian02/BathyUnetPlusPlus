from torch.utils.data import Dataset
import os
import glob
import pandas as pd
import numpy as np
import torch
from torchvision import transforms


class DepthData(Dataset):
    def __init__(self, features, targets, transform=None, 
                 is_norm=False, mean=None, std=None, 
                 min_vals=None, max_vals=None,
                 norm_type="zscore"):
        self.features = features
        self.targets = targets
        self.transform = transform
        self.is_norm = is_norm
        self.norm_type = norm_type
        if self.is_norm:
            # norm data didn't improve the bathymetry result, discard it
            if norm_type == "zscore":
                if mean is None or std is None:
                    raise ValueError("When is_norm=True, mean and std must be provided")
                # Convert mean and std to shape (1, -1, 1, 1)
                self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
                self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
            elif norm_type == "minmax":
                if min_vals is None or max_vals is None:
                    raise ValueError("When is_norm=True, min_vals and max_vals must be provided")
                self.min_vals = torch.tensor(min_vals, dtype=torch.float32).view(1, -1, 1, 1)
                self.max_vals = torch.tensor(max_vals, dtype=torch.float32).view(1, -1, 1, 1)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        if self.is_norm:
            if self.norm_type == "zscore":
                mean = self.mean.squeeze(0)
                std = self.std.squeeze(0)
                features = (features - mean) / std
            elif self.norm_type == "minmax":
                min_vals = self.min_vals.squeeze(0)
                max_vals = self.max_vals.squeeze(0)
                features = (features - min_vals) / (max_vals - min_vals)

        if self.transform:
            # To ensure features and targets get the same random transform, use same random state
            # Concatenate features and targets, transform together, then separate
            # For more transform options, use albumentations library
            # https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/
            if len(targets.shape) == 3:  # For patch2patch task, targets is (1, H, W)
                # Concatenate features and targets along channel dimension
                combined = torch.cat([features, targets], dim=0)  # (C+1, H, W)
                combined_transformed = self.transform(combined)
                features = combined_transformed[:features.shape[0]]  # First C channels
                targets = combined_transformed[features.shape[0]:]   # Last 1 channel
            else:
                # For other cases (e.g. single point regression), only transform features
                features = self.transform(features)
        
        return features, targets

    def __len__(self):
        return len(self.targets)

def cal_patch_stats(patches_paths, feature_key='patches_rhorc'):

    patches_files = []
    if isinstance(patches_paths, list):
        for path in patches_paths:
            patches_files.extend(glob.glob(os.path.join(path, "**/*.npz"), recursive=True))
    elif isinstance(patches_paths, str):
        patches_files = glob.glob(os.path.join(patches_paths, "**/*.npz"), recursive=True)

    channel_sums = None      # Σx
    channel_squared_sums = None  # Σx²
    channel_mins = None   
    channel_maxs = None   
    total_pixels = 0
    
    for path in patches_files:
        data = np.load(path)
        features = data[feature_key]  # (n, 32, 32, 8)
        
        # Convert to (n, 8, 32, 32) format, use float64 for better precision
        features = features.transpose(0, 3, 1, 2).astype(np.float64)
        
        if channel_sums is None:
            channel_sums = np.sum(features, axis=(0, 2, 3))
            channel_squared_sums = np.sum(features**2, axis=(0, 2, 3))
            channel_mins = np.min(features, axis=(0, 2, 3))
            channel_maxs = np.max(features, axis=(0, 2, 3))
        else:
            channel_sums += np.sum(features, axis=(0, 2, 3))
            channel_squared_sums += np.sum(features**2, axis=(0, 2, 3))
            batch_mins = np.min(features, axis=(0, 2, 3))
            batch_maxs = np.max(features, axis=(0, 2, 3))
            channel_mins = np.minimum(channel_mins, batch_mins)
            channel_maxs = np.maximum(channel_maxs, batch_maxs)
        
        total_pixels += features.shape[0] * features.shape[2] * features.shape[3]
    
    mean = channel_sums / total_pixels  # E[X]
    mean_squared = channel_squared_sums / total_pixels  # E[X²]
    variance = mean_squared - (mean ** 2)  # Var(X) = E[X²] - (E[X])²
    std = np.sqrt(variance)
    
    return mean.astype(np.float32), std.astype(np.float32), channel_mins.astype(np.float32), channel_maxs.astype(np.float32)

def cal_patch_stats_batch(patches_paths, feature_key='patches_rhorc'):

    patches_files = []
    if isinstance(patches_paths, list):
        for path in patches_paths:
            patches_files.extend(glob.glob(os.path.join(path, "**/*.npz"), recursive=True))
    elif isinstance(patches_paths, str):
        patches_files = glob.glob(os.path.join(patches_paths, "**/*.npz"), recursive=True)

    all_features = []
    for path in patches_files:
        data = np.load(path)
        features = data[feature_key]  # (n, 32, 32, 8)
        #  to (n, 8, 32, 32) 
        features = features.transpose(0, 3, 1, 2)
        all_features.append(features)
    
    # Merge all data (total_n, 8, 32, 32)
    all_features = np.concatenate(all_features, axis=0)
    
    # Use numpy to calculate mean and std for each channel
    # axis=(0, 2, 3) means compute over batch, height, width dimensions, keeping channel dimension
    mean = np.mean(all_features, axis=(0, 2, 3))  # shape: (8,)
    std = np.std(all_features, axis=(0, 2, 3))    # shape: (8,)
    minimum = np.min(all_features, axis=(0, 2, 3))  # shape: (8,)
    maximum = np.max(all_features, axis=(0, 2, 3))  # shape: (8,)
    
    return mean, std, minimum, maximum



def create_from_csv(csv_paths, feature_cols, target_col, mode="train", val_ratio=0.1):

    csv_files = []
    if isinstance(csv_paths, list):
        for path in csv_paths:
            csv_files.extend(glob.glob(os.path.join(path, "**/*.csv"), recursive=True))
    elif isinstance(csv_paths, str):
        csv_files = glob.glob(os.path.join(csv_paths, "**/*.csv"), recursive=True)

    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data = data[(data > 0).all(axis=1)]

    features = data[feature_cols].values.astype(np.float32)
    # !!! Must convert to (n,1), otherwise it becomes (n,) which causes loss computation issues
    targets = data[target_col].values.astype(np.float32).reshape(-1, 1)
    # delete nan
    flag_nan = ~(np.isnan(features).any(axis=1) | np.isnan(targets).any(axis=1))
    features = features[flag_nan, :]
    targets = targets[flag_nan, :]
    if (mode == "train") or (mode == "test"):
        return DepthData(features, targets)
    elif mode == "train/val":
        np.random.seed(42)
        idx = np.random.permutation(len(features))
        val_size = int(len(features) * val_ratio)
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]

        train_dataset = DepthData(features[train_idx], targets[train_idx])
        val_dataset = DepthData(features[val_idx], targets[val_idx])
        return train_dataset, val_dataset
    else:
        raise ValueError("Invalid mode")


def create_from_patches(patches_paths, feature_key='patches', target_key='depths',
                        mode="train", val_ratio=0.1):
    "input patch, output 1 pixel depth"
    if os.path.isfile(patches_paths):
        data = np.load(patches_paths)
        patches = data[feature_key]
        # !!! Must convert to (n,1), otherwise it becomes (n,) which causes loss computation issues
        depths = data[target_key].reshape(-1, 1)
        features = patches
        targets = depths

    else:
        patches_files = glob.glob(os.path.join(patches_paths, "**/*.npz"), recursive=True)
        patches_all = []
        depths_all = []
        for path in patches_files:
            data = np.load(path)
            patches_all.append(data[feature_key])
            # !!! Must convert to (n,1), otherwise it becomes (n,) which causes loss computation issues
            depths_all.append(data[target_key].reshape(-1, 1))
        features = np.concatenate(patches_all, axis=0)  # (n, 3, 3, 8)
        features = features.transpose(0, 3, 1, 2)  # (n, 8, 3, 3)
        targets = np.concatenate(depths_all, axis=0)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
        transforms.RandomVerticalFlip(p=0.5),  # Vertical flip
        transforms.RandomRotation(degrees=30),  # Random rotation
    ])

    if mode == "train":
        return DepthData(features, targets, transform=transform)
    elif mode == "test":
        return DepthData(features, targets)
    elif mode == "train/val":
        np.random.seed(42)
        idx = np.random.permutation(len(features))
        val_size = int(len(features) * val_ratio)
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]

        train_dataset = DepthData(features[train_idx], targets[train_idx], transform=transform)
        val_dataset = DepthData(features[val_idx], targets[val_idx])
        return train_dataset, val_dataset
    else:
        raise ValueError("Invalid mode")


def create_from_patch2patch(patches_paths, feature_key='patches_rhorc', target_key='patches_depth',
                            mode="train", val_ratio=0.1,
                            is_norm=False, mean=None, std=None, min_vals=None, max_vals=None, norm_type="zscore"):
    patches_files = []
    if isinstance(patches_paths, list):
        for path in patches_paths:
            patches_files.extend(glob.glob(os.path.join(path, "**/*.npz"), recursive=True))
    elif isinstance(patches_paths, str):
        patches_files = glob.glob(os.path.join(patches_paths, "**/*.npz"), recursive=True)

    patches_rhorc_all = []
    patches_depth_all = []
    for path in patches_files:
        data = np.load(path)
        patches_rhorc_all.append(data[feature_key])
        patches_depth_all.append(data[target_key])
    # (n, 32, 32, 8)
    features = np.concatenate(patches_rhorc_all, axis=0)
    # (n, 8, 32, 32)
    features = features.transpose(0, 3, 1, 2)
    targets = np.concatenate(patches_depth_all, axis=0)
    # (n, 1, 32, 32)
    targets = targets[:, np.newaxis, :, :]

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
        transforms.RandomVerticalFlip(p=0.5),  # Vertical flip
        transforms.RandomRotation(degrees=30),  # Random rotation
    ])

    if mode == "train":
        return DepthData(features, targets, transform=transform)
    elif mode == "test":
        return DepthData(features, targets)
    elif mode == "train/val":
        np.random.seed(42)
        idx = np.random.permutation(len(features))
        val_size = int(len(features) * val_ratio)
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]

        train_dataset = DepthData(features[train_idx], targets[train_idx], transform=transform, 
                                  is_norm=is_norm, mean=mean, std=std, min_vals=min_vals, max_vals=max_vals, norm_type=norm_type)
        val_dataset = DepthData(features[val_idx], targets[val_idx], 
                                is_norm=is_norm, mean=mean, std=std, min_vals=min_vals, max_vals=max_vals, norm_type=norm_type)
        return train_dataset, val_dataset
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    # bands_planet = [444, 492, 533, 566, 612, 666, 707, 866]
    # train_dataset, val_dataset = create_from_csv('/Users/wendy/Library/CloudStorage/OneDrive-个人/Code/Python/Planet/matchups_point_point/train',
    #                 [f"rhorc_{band}" for band in bands_planet],
    #                 'depth_ph',
    #                 mode="train/val")
    # print(train_dataset[0], val_dataset[0])

    # train_dataset, val_dataset = create_from_patches(
    #     r"D:\OneDrive\Code\Python\Planet\matchups_patch_centPoint\patch_101\matchups_Dongsha",
    #     feature_key='patches',
    #     target_key='depths',
    #     mode="train/val")
    # print(len(train_dataset))

    # load patch2patch data
    patches_paths = ["./data/matchups_patch_L8patch/train/patch32", 
                    "./data/matchups_patch_DEM/train/patch32"]
    train_dataset, val_dataset = create_from_patch2patch(
        patches_paths,
        feature_key='patches_rhorc',
        target_key='patches_depth',
        mode="train/val")
    print(len(train_dataset))
    print(train_dataset[0][0].shape, train_dataset[0][1].shape)
    print(len(val_dataset))

    # ============= cal mean and std =============
    # patches_paths = ["./data/matchups_patch_L8patch/train/patch32", 
    #                  "./data/matchups_patch_DEM/train/patch32"]
    # mean, std, minimum, maximum = cal_patch_stats(patches_paths)
    # print(mean, std, minimum, maximum)

    # mean, std, minimum, maximum = cal_patch_stats_batch(patches_paths)
    # print(mean, std, minimum, maximum)
    # ======================================= 

    # ============= normalize patch32 =============
    # # normalize zscore and minmax
    # mean_patch32 = [0.07258312, 0.08215376, 0.06976028, 0.0561446,  0.02278608, 0.01964586, 0.01455943, 0.01030613]
    # std_patch32 = [0.02049324, 0.02491514, 0.02960878, 0.03165695, 0.02303923, 0.01628718, 0.01252353, 0.00272418]
    # min_patch32 = [0.01785661, 0.02042111, 0.01478218, 0.00821646, 0.00311397, 0.00542143, 0.00226555, -0.00575114]
    # max_patch32 = [0.20369311, 0.2269628, 0.25883463, 0.24852556, 0.20766808, 0.18279758, 0.17776078, 0.01997923]

    # # load patch2patch data
    # train_dataset, val_dataset = create_from_patch2patch(
    #     patches_paths,
    #     feature_key='patches_rhorc',
    #     target_key='patches_depth',
    #     mode="train/val",
    #     is_norm=True,
    #     mean=mean_patch32,
    #     std=std_patch32,
    #     min_vals=min_patch32,
    #     max_vals=max_patch32,
    #     norm_type="minmax")
    # print(len(train_dataset))
    # print(train_dataset[0][0].shape, train_dataset[0][1].shape)
    # print(len(val_dataset))
    # ======================================= 