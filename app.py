import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import xarray as xr
import sys
import os
import importlib


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR, 'models'))
# from models.unetPlusPlus_attention import DepthUnetPlusPlus

def get_device(device_setting):
    if device_setting == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_setting)


def normalize_image(img, mean, std, min_vals=None, max_vals=None, norm_type="zscore"):
    """
    Normalize image using mean and std for each channel
    Args:
        img: numpy array of shape (C, H, W)
        mean: list of mean values for each channel
        std: list of std values for each channel
        min_vals: list of min values for each channel
        max_vals: list of max values for each channel
        norm_type: "zscore" or "minmax"
    Returns:
        normalized image
    """

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    if norm_type == "zscore":
        # Convert mean and std to numpy arrays and reshape to (C, 1, 1) for broadcasting
        mean_arr = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        std_arr = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
        img_norm = (img - mean_arr) / std_arr
    elif norm_type == "minmax":
        min_arr = np.array(min_vals, dtype=np.float32).reshape(-1, 1, 1)
        max_arr = np.array(max_vals, dtype=np.float32).reshape(-1, 1, 1)
        img_norm = (img - min_arr) / (max_arr - min_arr)

    return img_norm


def load_planet(fn, mask_l2flags=False):
    # don't mask l2_flags here, mask later after prediction, since we need full image for patch extraction
    # if there are many clouds, we should mask the clouds in this step
    with xr.open_dataset(fn) as ds:
        bands_planet = [444, 492, 533, 566, 612, 666, 707, 866]
        rhorc = [ds[f'rhorc_{band}'] for band in bands_planet]

        # create a new dimension 'band' (8,y,x), new dimension is at the front
        # y and x are rows and columns
        rhorc_xr = xr.concat(rhorc, dim='band')
        if mask_l2flags:
            l2_flags = ds['l2_flags'].values
            # cloud mask
            mask = (l2_flags != 0)
            rhorc_xr = rhorc_xr.where(~mask)

    return rhorc_xr


def save_nc(depth_pred, raw_ds, output_fn, only_output=True):
    if only_output:
        # Create new dataset with only depth and coordinates
        ds_output = xr.Dataset(
            data_vars={'predicted_depth': (('y', 'x'), depth_pred),
                       'lon': raw_ds['lon'],
                       'lat': raw_ds['lat'],
                       'l2_flags': raw_ds['l2_flags'],
                       'rhorc_866': raw_ds['rhorc_866']},
            coords={'y': raw_ds.y,
                    'x': raw_ds.x}
        )

        ## set beam dataformat global attributes
        ds_output.attrs = raw_ds.attrs
        ds_output.attrs['Conventions'] = 'CF-1.7'
        ds_output.attrs['product_type'] = 'NetCDF'
        ds_output.attrs['metadata_profile'] = 'beam'  # for seadas Reader Selection
        ds_output.attrs['generated_by'] = 'wendian'

        ds_output.to_netcdf(output_fn)
    else:
        # Save prediction back to original .nc file as a variable
        raw_ds['predicted_depth'] = (('y', 'x'), depth_pred)
        raw_ds.to_netcdf(output_fn)


def generate_patch_coords(img_shape, patch_size, stride=None):
    if stride is None:
        stride = patch_size // 2

    _, H, W = img_shape
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # upper-left corner
            coords.append((y, x))
    return coords


class InferenceDataset(Dataset):
    def __init__(self, image, coords, patch_size):
        super().__init__()
        self.image = image
        self.coords = coords
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch = self.image[:, y:y + self.patch_size, x:x + self.patch_size]
        return torch.from_numpy(patch).float()


def predict_sliding_window(model, image, coords, patch_size, batch_size=64, n_workers=0):
    dataset = InferenceDataset(image, coords, patch_size)

    patches_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    predictions = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(patches_loader):
            data = data.to(device)
            outputs = model(data)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    return predictions


def mosaicking_patches_weighted(predictions, coords, output_shape, patch_size):

    reconstructed = np.zeros(output_shape, dtype=np.float32)
    weight_map = np.zeros(output_shape, dtype=np.float32)

    # average
    weight = np.ones((patch_size, patch_size))

    # gaussian weight
    # y, x = np.mgrid[0:patch_size, 0:patch_size]
    # y = y - patch_size // 2
    # x = x - patch_size // 2
    # weight = np.exp(-(x ** 2 + y ** 2) / (patch_size ** 2))

    for i, (y, x) in enumerate(coords):
        # predictions-> (N, 1, H, W)
        pred = predictions[i, 0]
        # Add weighted prediction to the output
        reconstructed[y:y + patch_size, x:x + patch_size] += pred * weight
        weight_map[y:y + patch_size, x:x + patch_size] += weight
    # Normalize by the weight map
    mask = weight_map > 0
    reconstructed[mask] /= weight_map[mask]
    return reconstructed


def mosaicking_patches_center_crop(predictions, coords, output_shape,
                              patch_size, center_size=16):
    reconstructed = np.zeros(output_shape)
    weight_map = np.zeros(output_shape)
    weight = np.ones((center_size, center_size), dtype=np.float32)

    # Calculate start and end indices of center region
    start = (patch_size - center_size) // 2  # 8
    end = start + center_size  # 24

    # Generate gaussian weight
    # y, x = np.mgrid[0:center_size, 0:center_size]
    # y = y - center_size // 2
    # x = x - center_size // 2
    # weight = np.exp(-(x**2 + y**2) / (center_size**2))

    for i, (y0, x0) in enumerate(coords):
        # Extract prediction (H, W)
        pred_patch = predictions[i, 0]

        # Crop center region
        pred_center = pred_patch[start:end, start:end]

        # Calculate placement position in output image
        out_y0 = y0 + start  # y0+8
        out_x0 = x0 + start  # x0+8
        out_y1 = out_y0 + center_size  # y0+8+16
        out_x1 = out_x0 + center_size  # x0+8+16

        # Accumulate weighted results
        reconstructed[out_y0:out_y1, out_x0:out_x1] += pred_center * weight
        weight_map[out_y0:out_y1, out_x0:out_x1] += weight
        # reconstructed[out_y0:out_y1, out_x0:out_x1] += pred_center

    # Normalize
    mask = weight_map > 0
    reconstructed[mask] /= weight_map[mask]
    reconstructed[~mask] = np.nan  # set nan to non-covered areas[edge]

    # drop global margin
    reconstructed[:patch_size, :] = np.nan
    reconstructed[-patch_size:, :] = np.nan
    reconstructed[:, :patch_size] = np.nan
    reconstructed[:, -patch_size:] = np.nan

    return reconstructed


def main(model, input_fn, output_fn,
         patch_size=32,
         stride=None, centerPatch=True,
         batch_size=64, n_workers=0,
         is_norm=False,
         mean=None, std=None, min_vals=None, max_vals=None, norm_type="zscore",
         mask_l2flags=True,
         mask_rhorc_866=False,
         save_only_output=False):

    # don't mask l2_flags here, mask later after prediction
    img_3d_xr = load_planet(input_fn, mask_l2flags=False)  # (8,y,x)
    img_3d_arr = img_3d_xr.values

    # Norm
    img_3d_arr = normalize_image(img_3d_arr, mean, std, min_vals, max_vals, norm_type) if is_norm else img_3d_arr

    _, H, W = img_3d_arr.shape

    coords = generate_patch_coords(img_3d_arr.shape, patch_size, stride)
    predictions = predict_sliding_window(model, img_3d_arr, coords, patch_size, batch_size, n_workers)

    if centerPatch:
        output_depth = mosaicking_patches_center_crop(predictions, coords, (H, W), patch_size, center_size=patch_size // 2) # CVR
    else:
        output_depth = mosaicking_patches_weighted(predictions, coords, (H, W), patch_size)


    # remove l2_flags before saving
    with xr.open_dataset(input_fn) as ds:
        if mask_l2flags:
            l2_flags = ds['l2_flags'].values
            rhorc_866 = ds['rhorc_866'].values
            # mask flags
            keep_mask = ((l2_flags == 0) | (l2_flags == 8))
            # keep_mask = (l2_flags == 0)
            if mask_l2flags:
                output_depth[~keep_mask] = np.nan
            if mask_rhorc_866:
                output_depth[rhorc_866 > 0.02] = np.nan

        save_nc(output_depth, ds, output_fn, only_output=save_only_output)


if __name__ == "__main__":
    device = get_device('auto')
    print(f"Using device: {device}")

    model_name = "unetPlusPlus32_attention" # default
    # model_name = "unet32"
    # model_name = "unetPlusPlus32"
    # model_name = "unetPlusPlus64_attention"
    model = importlib.import_module(f"models.{model_name}")
    depth_model = model.get_model().to(device)

    model_params = 'BathyUnetPlusPlus_scSE.params'
    # model_params = 'unet32.params'
    # model_params = 'unet32PlusPlus.params'
    # model_params = 'P64.params'
    # model_params = 'only_LS.params'
    # model_params = 'only_HS.params'
    # model_params = 'standard_MSE.params'  # stride=patch_size//2, centerPatch=False

    depth_model.load_state_dict(torch.load(f"./save_model/{model_params}", map_location=device))

    patch_size = 32
    # patch_size = 64 

    # Overlapping sliding window to reduce edge effects
    stride = patch_size // 4 # CVR
    # stride = patch_size // 2 # standard MSE

    DATASET = {
    # "Dongsha": {
    #     "input_fn": "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/output/PlanetScope_24b4_2025_04_21_02_35_48_L2W.nc",
    #     "output_fn": f"/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_{model_params.split('.')[0]}.nc",
    # },
    # "Huangyan": {
    #     "input_fn": "/Volumes/DATA/Planet/Huangyan/2024-01-20/output/PlanetScope_24a4_2024_01_20_02_36_59_L2W.nc",
    #     "output_fn": f"/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_{model_params.split('.')[0]}.nc",
    # },
    # "Huaguang": {
    #     "input_fn": "/Volumes/DATA/Planet/Huaguang/2025-01-07/output/PlanetScope_24f4_2025_01_07_03_20_43_L2W.nc",
    #     "output_fn": f"/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_{model_params.split('.')[0]}.nc",
    # },
    # "Oahu": {
    #     "input_fn": "/Volumes/DATA/Planet/Oahu/2024-01-15/output/PlanetScope_24e5_2024_01_15_21_14_03_L2W.nc",
    #     "output_fn": f"/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_{model_params.split('.')[0]}.nc",
    # },
    # "Oahu_waveglint": {
    #     "input_fn": "/Volumes/DATA/Planet/Oahu/2024-07-22-wave-glint/output_rhorc/PlanetScope_24b3_2024_07_22_20_22_30_L2W.nc",
    #     "output_fn": f"/Volumes/DATA/Planet/Oahu/2024-07-22-wave-glint/depth_Oahu_20240722_{model_params.split('.')[0]}.nc",
    # },
    # "Huangyan_heavy_splash": {
    #     "input_fn": "/Volumes/DATA/Planet/Huangyan/2024-10-21-heavy-splash/output/PlanetScope_24c4_2024_10_21_02_07_38_L2W.nc",
    #     "output_fn": f"/Volumes/DATA/Planet/Huangyan/2024-10-21-heavy-splash/depth_Huangyan_20241021_{model_params.split('.')[0]}.nc",
    # },

    "Oahu_second": {
        "input_fn": "/Volumes/DATA/Planet/Oahu/2025-11-30/output/PlanetScope_24e9_2025_11_30_21_35_07_L2W.nc",
        "output_fn": f"/Volumes/DATA/Planet/Oahu/2025-11-30/depth_Oahu_20251130_{model_params.split('.')[0]}.nc",
    },
    }

    # for region, config in DATASET.items():
    #     input_fn = config["input_fn"]
    #     output_fn = config["output_fn"]
    #     main(
    #         model=depth_model,
    #         input_fn=input_fn,
    #         output_fn=output_fn,
    #         patch_size=patch_size,
    #         stride=stride,
    #         centerPatch=True, # True: CVR, False: standard MSE
    #         batch_size=128,
    #         n_workers=4,
    #         mask_l2flags=True,
    #         mask_rhorc_866=False, # for wave-glint and heavy-splash
    #         save_only_output=False,
    #     )
    
    import glob
    for full_path in glob.glob("/Volumes/DATA/Planet/Oahu/output/*/*/*L2W.nc"):
        in_fn = full_path
        fn = os.path.basename(in_fn)
        import re
        fn_date = re.search(r"PlanetScope_.*?_(\d{4}_\d{2}_\d{2})_\d{2}_\d{2}_\d{2}", fn).group(1)

        o_dir = os.path.dirname(os.path.dirname(in_fn))
        o_fn = f"depth_Oahu_{fn_date.replace('_', '')}_{model_params.split('.')[0]}.nc"
        o_fullpath = os.path.join(o_dir, o_fn)

        main(
            model=depth_model,
            input_fn=in_fn,
            output_fn=o_fullpath,
            patch_size=patch_size,
            stride=stride,
            centerPatch=True, # True: CVR, False: standard MSE
            batch_size=128,
            n_workers=4,
            mask_l2flags=True,
            mask_rhorc_866=False, # for wave-glint and heavy-splash
            save_only_output=False,
        )

