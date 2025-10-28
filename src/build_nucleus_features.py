"""
Minimal version of mask feature extraction.
Takes open Zarr arrays as input and returns a single DataFrame.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from skimage.measure import regionprops


# -----------------------------------------------------------------------------
# Core feature extraction
# -----------------------------------------------------------------------------

def extract_region_features(region) -> dict:
    """Compute geometric and intensity features for a single region."""
    vol = region.area
    convex_vol = getattr(region, "convex_area", np.nan)
    solidity = vol / convex_vol if convex_vol and convex_vol > 0 else np.nan
    extent = getattr(region, "extent", np.nan)

    mean_intensity = getattr(region, "mean_intensity", np.nan)
    min_intensity = getattr(region, "min_intensity", np.nan)
    max_intensity = getattr(region, "max_intensity", np.nan)

    eigvals = getattr(region, "inertia_tensor_eigvals", None)
    if eigvals is not None and len(eigvals) == 3 and eigvals[0] != 0:
        elongation_ratio = eigvals[-1] / eigvals[0]
    else:
        elongation_ratio = np.nan

    if hasattr(region, "bbox") and len(region.bbox) == 6:
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        bbox_z, bbox_y, bbox_x = max_z - min_z, max_y - min_y, max_x - min_x
    else:
        bbox_z = bbox_y = bbox_x = np.nan

    return dict(
        label=region.label,
        volume=vol,
        convex_volume=convex_vol,
        solidity=solidity,
        extent=extent,
        mean_intensity=mean_intensity,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        elongation_ratio=elongation_ratio,
        bbox_z=bbox_z,
        bbox_y=bbox_y,
        bbox_x=bbox_x,
    )


def process_frame(t: int, seg_zarr, img_zarr, scale_vec, nls_channel: int) -> pd.DataFrame:
    """Compute per-region features for a single timepoint and return a DataFrame."""
    seg = np.asarray(seg_zarr[t]).squeeze()
    img = np.asarray(img_zarr[t, nls_channel]).squeeze()
    regions = regionprops(seg, intensity_image=img, spacing=scale_vec)
    df = pd.DataFrame([extract_region_features(r) for r in regions])
    df["frame"] = t
    return df


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_mask_features(
    seg_zarr,
    img_zarr,
    scale_vec,
    nls_channel: int,
    *,
    start_i: int = 0,
    stop_i: int | None = None,
    parallel: bool = False,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Build per-mask feature table from given segmentation and image Zarrs.

    Parameters
    ----------
    seg_zarr : zarr.Array
        Label mask array (T, Z, Y, X).
    img_zarr : zarr.Array
        Raw image array (T, C, Z, Y, X) or (T, C, Y, X).
    scale_vec : sequence of float
        Physical pixel size in µm (Z, Y, X).
    nls_channel : int
        Channel index of the nuclear signal.
    start_i, stop_i : int
        Frame range to process (defaults to all frames).
    parallel : bool
        Use process_map for parallel execution.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    pd.DataFrame
        Combined feature table with columns:
        ['label', 'volume', 'convex_volume', 'solidity', 'extent',
         'mean_intensity', 'min_intensity', 'max_intensity',
         'elongation_ratio', 'bbox_z', 'bbox_y', 'bbox_x', 'frame']
    """
    stop_i = stop_i or seg_zarr.shape[0]
    frames = range(start_i, stop_i)

    if parallel:
        from multiprocessing import cpu_count
        if n_workers is None:
            n_workers = max(1, cpu_count() // 3)
        dfs = process_map(
            lambda t: process_frame(t, seg_zarr, img_zarr, scale_vec, nls_channel),
            frames,
            max_workers=n_workers,
            chunksize=1,
            desc="Extracting mask features",
            unit="frame",
        )
    else:
        dfs = [process_frame(t, seg_zarr, img_zarr, scale_vec, nls_channel) for t in tqdm(frames)]

    feature_df = pd.concat(dfs, ignore_index=True)
    return feature_df
