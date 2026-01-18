import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import xarray as xr
# from Planet.models.mlp import MLPDepth
import importlib

def get_device(device_setting="auto"):
    if device_setting == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_setting)

def xr_read_planet(fn, mask_l2flags=False):
    with xr.open_dataset(fn) as ds:
        bands_planet = [444, 492, 533, 566, 612, 666, 707, 866]
        rhorc = [ds[f'rhorc_{band}'] for band in bands_planet]

        # create a new dimension 'band' (8,y,x), new dimension is at the front
        # y and x are rows and columns
        rhorc_xr = xr.concat(rhorc, dim='band')

        if mask_l2flags:
            l2_flags = ds['l2_flags'].values
            # Merge x and y dimensions into multi-index 'points' -> (xy,8)
            rhorc_865 = rhorc_xr.values[-1, :, :]

            # cloud mask
            keep = ((l2_flags == 0) | (l2_flags == 8))
            # keep = ((l2_flags == 0) | (l2_flags == 8)) & (rhorc_865 < 0.02) # mask 865 < 0.02
            rhorc_xr = rhorc_xr.where(keep)
    rhorc_xr = rhorc_xr.stack(points=('y', 'x')).transpose('points', 'band')     # ->（xy, nband）
    return rhorc_xr

def predict(model, feature, batch_size=10000):
    features = torch.tensor(feature, dtype=torch.float32)
    features_loader = DataLoader(features, batch_size=batch_size, shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(features_loader):
            data = data.to(device)
            outputs = model(data)
            predictions.append(outputs.cpu().numpy())
        # (n,1)
        predictions = np.concatenate(predictions, axis=0)
    return predictions

def save_nc(depth_pred, raw_ds, output_fn, only_output=True):
    if only_output:
        # Create new dataset with only depth and coordinates
        ds_output = xr.Dataset(
            data_vars={'predicted_depth': depth_pred,
                       'lon': raw_ds['lon'],
                       'lat': raw_ds['lat']},
            coords={'y': raw_ds.y,
                    'x': raw_ds.x}
        )
        ds_output.to_netcdf(output_fn)
    else:
        raw_ds['predicted_depth'] = depth_pred
        raw_ds.to_netcdf(output_fn)

def main(model, input_fn, output_fn):
    rhorc_xr = xr_read_planet(input_fn, mask_l2flags=True) # distinct from app for unet, it can mask clouds here
    # (n,8)
    features = rhorc_xr.values
    valid_idx = ~np.isnan(features).any(axis=1)
    valid_features = features[valid_idx]

    # inference (n,1)
    predictions = predict(model, valid_features, batch_size=4096)

    full_predictions = np.full(rhorc_xr.points.size, np.nan)
    full_predictions[valid_idx] = predictions.flatten()
    depth_pred = xr.DataArray(full_predictions,
                              coords={'points': rhorc_xr.points},
                              dims=["points"])
    depth_pred = depth_pred.unstack('points')

    # save
    with xr.open_dataset(input_fn) as ds:
        save_nc(depth_pred, ds, output_fn, only_output=False)

if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    # load model
    model_name = "mlp"
    model = importlib.import_module(f"models.{model_name}")
    depth_model = model.get_model().to(device)
    model_params = f"mlp.params"
    depth_model.load_state_dict(torch.load(f"./save_model/{model_params}", map_location=device))


    DATASET = {
        # 'Dongsha': {
        #     'input_fn': "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/output/PlanetScope_24b4_2025_04_21_02_35_48_L2W.nc",
        #     'output_fn': f"/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_{model_name}.nc"
        # },
        # 'Huangyan': {
        #     'input_fn': "/Volumes/DATA/Planet/Huangyan/2024-01-20/output/PlanetScope_24a4_2024_01_20_02_36_59_L2W.nc",
        #     'output_fn': f"/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_{model_name}.nc"
        # },
        # 'Huaguang': {
        #     'input_fn': "/Volumes/DATA/Planet/Huaguang/2025-01-07/output/PlanetScope_24f4_2025_01_07_03_20_43_L2W.nc",
        #     'output_fn': f"/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_{model_params.split('.')[0]}.nc"
        # },
        # 'Oahu': {
        #     'input_fn': "/Volumes/DATA/Planet/Oahu/2024-01-15/output/PlanetScope_24e5_2024_01_15_21_14_03_L2W.nc",
        #     'output_fn': f"/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_{model_params.split('.')[0]}.nc"
        # },
        # 'Beidao': {
        #     'input_fn': "/Volumes/DATA/Planet/Beidao/2024-01-15/output/PlanetScope_24e5_2024_01_15_21_14_03_L2W.nc",
        #     'output_fn': f"/Volumes/DATA/Planet/Beidao/2024-01-15/depth_Beidao_20240115_{model_params.split('.')[0]}.nc"
        # },
        "Huangyan_heavy_splash": {
            'input_fn': "/Volumes/DATA/Planet/Huangyan/2024-10-21-heavy-splash/output/PlanetScope_24c4_2024_10_21_02_07_38_L2W.nc",
            'output_fn': f"/Volumes/DATA/Planet/Huangyan/2024-10-21-heavy-splash/depth_Huangyan_20241021_{model_params.split('.')[0]}.nc"
        },
    }
    for region, config in tqdm(DATASET.items()):
        input_fn = config['input_fn']
        output_fn = config['output_fn']
        main(depth_model, input_fn, output_fn)