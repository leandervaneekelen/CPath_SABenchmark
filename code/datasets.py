import os
import pandas as pd
import torch
import numpy as np
from pathlib import Path


def get_datasets(mccv, data, y_label, encoder, method):
    df = pd.read_csv(data)
    df_train = (
        df.loc[df[f"mccv{mccv}"] == "train", [encoder, y_label]]
        .reset_index(drop=True)
        .dropna()
        .rename(columns={y_label: "y"})
    )
    df_val = (
        df.loc[df[f"mccv{mccv}"] == "val", [encoder, y_label]]
        .reset_index(drop=True)
        .dropna()
        .rename(columns={y_label: "y"})
    )
    df_test = None

    if method in [
        "GTP",
        "PatchGCN",
        "DeepGraphConv",
        "MIL_Cluster_FC",
        "MIL_Sum_FC",
        "ViT_MIL",
        "DTMIL",
    ]:
        dset_train = slide_dataset_classification_graph(df_train)
        dset_val = slide_dataset_classification_graph(df_val)
        dset_test = (
            slide_dataset_classification_graph(df_test) if df_test is not None else None
        )
    else:
        dset_train = slide_dataset_classification(df_train)
        dset_val = slide_dataset_classification(df_val)
        dset_test = (
            slide_dataset_classification(df_test) if df_test is not None else None
        )

    return dset_train, dset_val, dset_test


class slide_dataset_classification(torch.utils.data.Dataset):
    """
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    """

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path_to_data, label = self.df.iloc[index]
        data = np.load(path_to_data)  # feature matrix and possibly other data
        try:
            feat = data["features"]
        except:
            feat = data
        return {"features": feat, "target": label}


class slide_dataset_classification_graph(slide_dataset_classification):
    def __init__(self, df):
        super(slide_dataset_classification_graph, self).__init__(df)

    def __getitem__(self, index):
        # Load data using the parent class method
        item = super(slide_dataset_classification_graph, self).__getitem__(index)
        # Additional graph-specific data extraction
        data = torch.load(self.df.iloc[index].method_tensor_path)
        if "adj_mtx" in data:  # GTP
            item["adj_mtx"] = data["adj_mtx"]
            item["mask"] = data["mask"]
        if "edge_latent" in data.keys():  # PatchGCN, DeepGraphConv
            item["edge_index"] = data["edge_index"]
            item["edge_latent"] = data["edge_latent"]
            item["centroid"] = data["centroid"]
        if "feat_map" in data:  # ViT_MIL, DTMIL
            item["feat_map"] = data["feat_map"]
            item["mask"] = data["mask"]
        return item
