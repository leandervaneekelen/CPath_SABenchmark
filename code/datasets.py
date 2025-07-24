import pandas as pd
import torch
import numpy as np


def get_survival_datasets(mccv, data, y_label, encoder, method, tile_index=None):
    df = pd.read_csv(data).rename(
        columns={encoder: "encoder", y_label: "y", tile_index: "tile_index_path"}
    )
    columns = ["encoder", "y", "discrete_label", "censored"]
    if tile_index is not None:
        columns.append("tile_index_path")

    df_train = df.loc[df[f"mccv{mccv}"] == "train", columns].reset_index(drop=True)
    df_val = df.loc[df[f"mccv{mccv}"] == "val", columns].reset_index(drop=True)
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
        dset_train = slide_dataset_survival_graph(df_train)
        dset_val = slide_dataset_survival_graph(df_val)
        dset_test = (
            slide_dataset_survival_graph(df_test) if df_test is not None else None
        )
    else:
        dset_train = slide_dataset_survival(df_train)
        dset_val = slide_dataset_survival(df_val)
        dset_test = slide_dataset_survival(df_test) if df_test is not None else None

    return dset_train, dset_val, dset_test


class slide_dataset_survival(torch.utils.data.Dataset):
    """
    Slide level dataset which returns for each slide the feature matrix (h) and the discretized label, time-to-event and censoring status.
    """

    def __init__(self, df):
        self.df = df
        self.n_bins = df.discrete_label.nunique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path_to_data = row["encoder"]
        time_to_event = row["y"]
        discrete_label = row["discrete_label"]
        censored = row["censored"]

        data = np.load(path_to_data)  # feature matrix and possibly other data
        try:
            feat = data["features"]
        except:
            feat = data

        if "tile_index" in row.keys():
            tile_index = np.load(row["tile_index"])
            feat = feat[tile_index]

        return {
            "features": feat,
            "discrete_label": discrete_label,
            "time_to_event": time_to_event,
            "censored": censored,
        }


# TODO: implement this class
class slide_dataset_survival_graph(slide_dataset_survival):
    def __init__(self, df):
        super(slide_dataset_survival_graph, self).__init__(df)

    def __getitem__(self, index):
        # Load data using the parent class method
        item = super(slide_dataset_survival_graph, self).__getitem__(index)
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


def get_classification_datasets(mccv, data, y_label, encoder, method, tile_index=None):
    df = pd.read_csv(data)
    columns = [encoder, y_label]
    if tile_index is not None:
        columns.append(tile_index)
    rename = {encoder: "encoder", y_label: "y", tile_index: "tile_index"}
    df_train = (
        df.loc[df[f"mccv{mccv}_{y_label}"] == "train", columns]
        .reset_index(drop=True)
        .dropna()
        .rename(rename)
    )
    df_val = (
        df.loc[df[f"mccv{mccv}_{y_label}"] == "val", columns]
        .reset_index(drop=True)
        .dropna()
        .rename(rename)
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
        row = self.df.iloc[index]
        path_to_data = row["encoder"]
        label = row["y"]
        data = np.load(path_to_data)  # feature matrix and possibly other data
        try:
            feat = data["features"]
        except:
            feat = data

        if "tile_index" in row.keys():
            tile_index = np.load(row["tile_index"])
            feat = feat[tile_index]
        return {"features": feat, "target": label}


# TODO: implement this class
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
