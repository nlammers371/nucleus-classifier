import zarr
import napari
import numpy as np
from pathlib import Path

# load zarr image and mask data
mask_path = Path(r"./data/sample_masks.zarr")
mask_zarr = zarr.open(mask_path, mode="r")
image_path = Path(r"./data/sample_images.zarr")
image_zarr = zarr.open(image_path, mode="r")

nucleus_channel = 1
# get data scale
scale_vec = image_zarr.attrs.get("voxel_size_um", None)

# plot first 2 frames
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(image_zarr[0:2, nucleus_channel], scale=scale_vec)
viewer.add_labels(mask_zarr[0:2], scale=scale_vec)

# what if we colored labels by nucleus type?

if __name__ == '__main__':
    napari.run()
