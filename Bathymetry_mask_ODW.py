import os.path
import pathlib
import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import xy
import pyproj
from tqdm import tqdm

def read_L8_water_type(fn):
    # read predicted depth
    with xr.open_dataset(fn) as ds:
        lon = ds['lon'].values
        lat = ds['lat'].values
        depth = ds['depth'].values
        l2_flags = ds['l2_flags'].values
        water_type = ds['water_type'].values
        # cloud mask, 866 < 0.02
        mask = (l2_flags != 0)
        depth[mask] = np.nan

    return lon, lat, depth, water_type


def read_S2_water_type(tiff_fn, L2W_fn):
    # Read GeoTIFF file
    with rasterio.open(tiff_fn) as src:
        water_type = src.read(1).astype("float32")
        profile = src.profile  # Metadata (projection, resolution, band count, etc.)
        transform = src.transform  # Affine transform matrix
        crs = src.crs  # Coordinate reference system
        nodata = src.nodata
        height, width = src.height, src.width

    # NoData
    if nodata is not None:
        water_type = np.where(water_type == nodata, np.nan, water_type)
    water_type = water_type / 100  # normalize to 0-1

    # mask l2flags
    with xr.open_dataset(L2W_fn) as ds:
        l2_flags = ds['l2_flags'].values
    water_type[(l2_flags != 0) & (l2_flags != 8)] = np.nan  # cloud mask using l2_flags

    # coordinates
    cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
    x, y = map(np.asarray, xy(src.transform, rows, cols, offset="center"))

    # project
    wgs84 = pyproj.CRS.from_epsg(4326)
    if crs and not crs.is_geographic:
        lon, lat = pyproj.Transformer.from_crs(crs, wgs84, always_xy=True).transform(x, y)
    else:
        lon, lat = x, y

    return lon, lat, water_type


def read_planet_pre_depth(fn, mask_rhorc865=False, thre=0.02):
    # read predicted depth
    with xr.open_dataset(fn) as ds:
        lon = ds['lon'].values
        lat = ds['lat'].values
        depth = ds['predicted_depth'].values
        if mask_rhorc865:
            rhorc_866 = ds['rhorc_866'].values
            depth[rhorc_866 > thre] = np.nan  # mask land
    return lon, lat, depth


def resample_low2high(lon_low, lat_low, array_low, lon_high, lat_high, radius=30):
    from pyresample import geometry, kd_tree
    source_geo = geometry.SwathDefinition(lons=lon_low, lats=lat_low)
    target_geo = geometry.SwathDefinition(lons=lon_high, lats=lat_high)

    # Resample depth data to Planet pixel locations
    resampled_depth = kd_tree.resample_nearest(
        source_geo, array_low, target_geo,
        radius_of_influence=radius,  # 30 meters
        fill_value=np.nan
    )
    return resampled_depth

def load_wt_config(region):
    wt_config = {
        'Dongsha': {
            'S2_fn_tiff': '/Volumes/DATA/Sentinel-2/Dongsha/2025-08-07/output-osw/S2A_MSI_2025_08_07_02_41_43_50QMH_L2R_GEE_2025-08-07_OSW_ODW.tif',
            'S2_fn_L2W': '/Volumes/DATA/Sentinel-2/Dongsha/2025-08-07/S2A_MSI_2025_08_07_02_41_43_50QMH_L2W_GEE_2025-08-07.nc',
            'L8_fn': '/Volumes/DATA/Landsat/Dongsha/2025-05-01-train/L8_OLI_2025_05_01_02_33_47_119046_NNTOAB_GEE_2025-05-01.nc',
        },
        'Huaguang': {
            'S2_fn_tiff': '/Volumes/DATA/Sentinel-2/Huaguang/2025-01-23/output-osw/S2C_MSI_2025_01_23_03_11_17_49PET_L2R_GEE_2025-01-23_OSW_ODW.tif',
            'S2_fn_L2W': '/Volumes/DATA/Sentinel-2/Huaguang/2025-01-23/S2C_MSI_2025_01_23_03_11_17_49PET_L2W_GEE_2025-01-23.nc',
            'L8_fn': '/Volumes/DATA/Landsat/Huaguang/2025-03-03-sel/L8_OLI_2025_03_03_02_53_57_122049_NNTOAB_GEE_2025-03-03.nc',
        },
        'Huangyan': {
            'S2_fn_tiff': '/Volumes/DATA/Sentinel-2/Huangyan/2024-02-24-sel/output-osw/S2B_MSI_2024_02_24_02_39_32_50PNB_L2R_GEE_2024-02-24_OSW_ODW.tif',
            'S2_fn_L2W': '/Volumes/DATA/Sentinel-2/Huangyan/2024-02-24-sel/S2B_MSI_2024_02_24_02_39_32_50PNB_L2W_GEE_2024-02-24.nc',
            'L8_fn': '/Volumes/DATA/Landsat/Huangyan/2024-08-19-sel/L9_OLI_2024_08_19_02_29_04_118049_NNTOAB_GEE_2024-08-19.nc',
        },
        'Oahu': {
            'S2_fn_tiff': '/Volumes/DATA/Sentinel-2/Oahu/2025-02-18-sel/output-osw/S2B_MSI_2025_02_18_21_09_26_04QFJ_L2R_GEE_2025-02-18_OSW_ODW.tif',
            'S2_fn_L2W': '/Volumes/DATA/Sentinel-2/Oahu/2025-02-18-sel/S2B_MSI_2025_02_18_21_09_26_04QFJ_L2W_GEE_2025-02-18.nc',
            'L8_fn': '/Volumes/DATA/Landsat/Oahu/2024-12-22-sel/L8_OLI_2024_12_22_20_54_06_064045_NNTOAB_GEE_2024-12-22.nc',
        },
        'Saipan': {
            'S2_fn_tiff': '/Volumes/DATA/Sentinel-2/Saipan/2024-10-13/output-osw/S2A_MSI_2024_10_13_00_52_00_55PCS_L2R_GEE_2024-10-13_OSW_ODW.tif',
            'S2_fn_L2W': '/Volumes/DATA/Sentinel-2/Saipan/2024-10-13/S2A_MSI_2024_10_13_00_52_00_55PCS_L2W_GEE_2024-10-13.nc',

        },
        'Anegada': {
            'S2_fn_tiff': '/Volumes/DATA/Sentinel-2/Anegada/2025-01-18/output-osw/S2B_MSI_2025_01_18_14_57_23_20QLF_L2R_GEE_2025-01-18_OSW_ODW.tif',
            'S2_fn_L2W': '/Volumes/DATA/Sentinel-2/Anegada/2025-01-18/S2B_MSI_2025_01_18_14_57_23_20QLF_L2W_GEE_2025-01-18.nc',
        },
    }
    return wt_config[region]

if __name__ == '__main__':

    DATASET_SD = {
        'Oahu': [
            # "/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_BathyUnetPlusPlus_scSE.nc",
            # "/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_unet32.nc",
            # "/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_unet32PlusPlus.nc",
            # "/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_P64.nc",
            # "/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_only_HS.nc",
            # "/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_only_LS.nc",
            # "/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_standard_MSE.nc",
            # "/Volumes/DATA/Planet/Oahu/2024-01-15/depth_Oahu_20240115_mlp.nc",
            # "/Volumes/DATA/Planet/Oahu/2024-07-22-wave-glint/depth_Oahu_20240722_BathyUnetPlusPlus_scSE.nc", # case

            "/Volumes/DATA/Planet/Oahu/output/2024-11-18/depth_Oahu_20241118_BathyUnetPlusPlus_scSE.nc",
            "/Volumes/DATA/Planet/Oahu/output/2024-11-21/depth_Oahu_20241121_BathyUnetPlusPlus_scSE.nc",
            "/Volumes/DATA/Planet/Oahu/output/2025-02-11/depth_Oahu_20250211_BathyUnetPlusPlus_scSE.nc",
            "/Volumes/DATA/Planet/Oahu/output/2025-02-25/depth_Oahu_20250225_BathyUnetPlusPlus_scSE.nc",
            "/Volumes/DATA/Planet/Oahu/output/2025-03-15/depth_Oahu_20250315_BathyUnetPlusPlus_scSE.nc",
            "/Volumes/DATA/Planet/Oahu/output/2025-09-29/depth_Oahu_20250929_BathyUnetPlusPlus_scSE.nc",
            "/Volumes/DATA/Planet/Oahu/output/2024-12-16/depth_Oahu_20241216_BathyUnetPlusPlus_scSE.nc",
            
        ],
        'Dongsha': [
            # "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_BathyUnetPlusPlus_scSE.nc",
            # "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_unet32.nc",
            # "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_unet32PlusPlus.nc",
            # "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_P64.nc",
            # "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_only_HS.nc",
            # "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_only_LS.nc",
            # "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_standard_MSE.nc",
            # "/Volumes/DATA/Planet/Dongsha/2025-04-21-clear-test/depth_Dongsha_20250421_mlp.nc"
        ],
        'Huaguang': [
            # "/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_BathyUnetPlusPlus_scSE.nc",
            # "/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_unet32.nc",
            # "/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_unet32PlusPlus.nc",
            # "/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_P64.nc",
            # "/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_only_HS.nc",
            # "/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_only_LS.nc",
            # "/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_standard_MSE.nc",
            # "/Volumes/DATA/Planet/Huaguang/2025-01-07/depth_Huaguang_20250107_mlp.nc"
        ],
        'Huangyan': [
            # "/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_BathyUnetPlusPlus_scSE.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_unet32.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_unet32PlusPlus.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_P64.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_only_HS.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_only_LS.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_standard_MSE.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-01-20/depth_Huangyan_20240120_mlp.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-10-21-heavy-splash/depth_Huangyan_20241021_BathyUnetPlusPlus_scSE.nc", # case
            # "/Volumes/DATA/Planet/Huangyan/2024-10-21-heavy-splash/depth_Huangyan_20241021_P64.nc",
            # "/Volumes/DATA/Planet/Huangyan/2024-10-21-heavy-splash/depth_Huangyan_20241021_mlp.nc",
        ],
        # 'Saipan': [
        #     "/Volumes/DATA/Planet/Saipan/2025-12-10/depth_Saipan_20251210_unetPlusPlus32_attention.nc",
        # ],
        # 'Anegada': [
        #     "/Volumes/DATA/Planet/Anegada/depth_Anegada_20241216_unetPlusPlus32_attention.nc",
        # ]
    }

    for REGION, fns in tqdm(DATASET_SD.items(), desc="Processing..."):

        # REGION = 'Oahu'
        wt_config = load_wt_config(REGION)

        lon_l8, lat_l8, _, water_type_l8 = read_L8_water_type(wt_config['L8_fn']) if 'L8_fn' in wt_config else None, None, None, None
        lon_S2, lat_S2, water_type_S2 = read_S2_water_type(wt_config['S2_fn_tiff'], wt_config['S2_fn_L2W']) if ('S2_fn_tiff' in wt_config and 'S2_fn_L2W' in wt_config) else (None, None, None)

        for SD_fn in fns:
            SD_fn = pathlib.Path(SD_fn)
            lon_planet, lat_planet, depth_planet = read_planet_pre_depth(SD_fn, mask_rhorc865=False)

            # water_type_resl8 = resample_low2high(lon_l8, lat_l8, water_type_l8, lon_planet, lat_planet, radius=30)
            # depth_planet_mask_ODW_L8 = depth_planet.copy()
            # depth_planet_mask_ODW_L8[(water_type_resl8 > 0.5) | np.isnan(water_type_resl8)] = np.nan  # Keep only water pixels

            # s2 odw mask
            water_type_resS2 = resample_low2high(lon_S2, lat_S2, water_type_S2, lon_planet, lat_planet, radius=10)
            depth_planet_mask_ODW_S2 = depth_planet.copy()
            depth_planet_mask_ODW_S2[(water_type_resS2 < 0.6) | np.isnan(water_type_resS2)] = np.nan  # Keep only water pixels

            with xr.open_dataset(SD_fn) as ds:
                # ds['predicted_depth_mask_ODW_L8'] = (('y', 'x'), depth_planet_mask_ODW_L8)
                ds['predicted_depth_mask_ODW_S2'] = (('y', 'x'), depth_planet_mask_ODW_S2)
                output_fn = os.path.join(
                    SD_fn.parent,
                    SD_fn.stem + '_mask_ODW.nc')

                ds.to_netcdf(output_fn)
