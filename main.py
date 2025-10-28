from dev.src.build_nucleus_features import build_mask_features
from pathlib import Path
import zarr
import pandas as pd
from joblib import load

# sample script to generate nucleus mask features and apply a pre-trained RF classifier

if __name__ == "__main__":

    # load zarr image and mask data
    mask_path = Path(r"./data/sample_masks.zarr")
    mask_zarr = zarr.open(mask_path, mode="r")
    image_path = Path(r"./data/sample_images.zarr")
    image_zarr = zarr.open(image_path, mode="r")

    # get data scale
    scale_vec = image_zarr.attrs.get("voxel_size_um", None)

    # build mask feature table
    feature_df = build_mask_features(mask_zarr,
                                     image_zarr,
                                     nls_channel=1,
                                     scale_vec=scale_vec)

    # save
    feature_df.to_csv("./data/nucelus_features.csv", index=False)

    # apply classifier trained on (questionably) labeled dataset
    # load mask features
    # feature_df = pd.read_csv("./rf_classifier/nucelus_features.csv")

    # load classifier
    clf = load("./rf_classifier/nucleus_rf_classifier.joblib")

    print("Getting predictions....")
    # strip to just features for prediction
    X = feature_df.drop(columns=["frame", "label"], axis=1)
    Y = clf.predict(X)
    logL = clf.predict_log_proba(X)

    print("Check")

