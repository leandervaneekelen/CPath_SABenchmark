import pandas as pd
import torch
import numpy as np

from sklearn.utils.class_weight import compute_class_weight


def get_survival_datasets(
    fold,
    data,
    y_label,
    encoder,
    method,
    tile_index=None,
    time_to_event="Overall survival",
    event_label="Deceased",
    n_subsamples=None,
    random_seed=None,
    noise_std=None,
    cache_in_memory=False,
):

    df = pd.read_csv(data)
    rename = {
        encoder: "encoder",
        y_label: "y",
        tile_index: "tile_index",
        time_to_event: "time_to_event",
        event_label: "event_label",
    }
    df = df.rename(columns=rename)

    columns = ["encoder", "y", "censored", "time_to_event", "event_label"]
    if tile_index is not None:
        columns.append("tile_index")

    df_train = df.loc[df[fold] == "train", columns].reset_index(drop=True)
    df_val = df.loc[df[fold] == "val", columns].reset_index(drop=True)
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
        dset_train = slide_dataset_survival_graph(
            df_train,
            n_subsamples=n_subsamples,
            random_seed=random_seed,
            noise_std=noise_std,
            cache_in_memory=cache_in_memory,
        )
        dset_val = slide_dataset_survival_graph(
            df_val, cache_in_memory=cache_in_memory
        )  # Cache validation but no augmentations
        dset_test = (
            slide_dataset_survival_graph(df_test, cache_in_memory=cache_in_memory)
            if df_test is not None
            else None
        )
    else:
        dset_train = slide_dataset_survival(
            df_train,
            n_subsamples=n_subsamples,
            random_seed=random_seed,
            noise_std=noise_std,
            cache_in_memory=cache_in_memory,
        )
        dset_val = slide_dataset_survival(
            df_val, cache_in_memory=cache_in_memory
        )  # Cache validation but no augmentations
        dset_test = (
            slide_dataset_survival(df_test, cache_in_memory=cache_in_memory)
            if df_test is not None
            else None
        )

    return dset_train, dset_val, dset_test


class slide_dataset_survival(torch.utils.data.Dataset):
    """
    Slide level dataset which returns for each slide the feature matrix (h) and the discretized label, time-to-event and censoring status.
    """

    def __init__(
        self,
        df,
        n_subsamples=None,
        random_seed=None,
        noise_std=None,
        cache_in_memory=False,
    ):
        self.df = df
        self.n_bins = df.y.nunique()
        self.n_subsamples = n_subsamples
        self.random_seed = random_seed
        self.noise_std = noise_std  # Standard deviation for Gaussian noise
        self.cache_in_memory = cache_in_memory
        self.cached_features = {}  # Dictionary to store cached features

        if random_seed is not None:
            np.random.seed(random_seed)

        # Cache all features in memory if requested
        if self.cache_in_memory:
            print(f"Caching {len(self.df)} samples in memory...")
            self._cache_all_features()
            print("Caching completed!")

    def _cache_all_features(self):
        """Load all features into memory"""
        for idx, row in self.df.iterrows():
            path_to_data = row["encoder"]

            # Load the feature data
            data = np.load(path_to_data)
            try:
                feat = data["features"]
            except:
                feat = data

            # Apply tile index if available
            if "tile_index" in row.keys():
                if not pd.isna(row["tile_index"]):
                    tile_index = np.load(row["tile_index"])
                    feat = feat[tile_index]

            # Store in cache using the dataframe index as key
            self.cached_features[idx] = feat

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        discrete_label = row["y"]
        time_to_event = row["time_to_event"]
        censored = bool(row["censored"])

        # Get features from cache or load from disk
        if self.cache_in_memory:
            # Get the actual dataframe index (not the iloc index)
            df_index = self.df.index[index]
            selected_feat = self.cached_features[
                df_index
            ].copy()  # Copy to avoid modifying cached data
        else:
            # Load from disk (original behavior)
            path_to_data = row["encoder"]
            data = np.load(path_to_data)
            try:
                feat = data["features"]
            except:
                feat = data

            if "tile_index" in row.keys():
                if not pd.isna(row["tile_index"]):
                    tile_index = np.load(row["tile_index"])
                    feat = feat[tile_index]
            selected_feat = feat

        # Apply random subsampling if specified
        if self.n_subsamples is not None and len(selected_feat) > 0:
            n_tiles = len(selected_feat)
            if n_tiles <= self.n_subsamples:
                # If we have fewer tiles than requested, use all tiles
                pass  # selected_feat remains unchanged
            else:
                # Randomly sample n_subsamples tiles
                selected_indices = np.random.choice(
                    n_tiles, size=self.n_subsamples, replace=False
                )
                selected_feat = selected_feat[selected_indices]

        # Apply Gaussian noise augmentation if specified
        if self.noise_std is not None and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, selected_feat.shape)
            selected_feat = selected_feat + noise.astype(selected_feat.dtype)

        return {
            "features": selected_feat,
            "target": discrete_label,
            "time_to_event": time_to_event,
            "censored": censored,
        }


# TODO: implement this class
class slide_dataset_survival_graph(slide_dataset_survival):
    def __init__(
        self,
        df,
        n_subsamples=None,
        random_seed=None,
        noise_std=None,
        cache_in_memory=False,
    ):
        super(slide_dataset_survival_graph, self).__init__(
            df, n_subsamples, random_seed, noise_std, cache_in_memory
        )

    def __getitem__(self, index):
        # Load data using the parent class method (which includes caching, subsampling and noise)
        item = super(slide_dataset_survival_graph, self).__getitem__(index)

        # Additional graph-specific data extraction (these are typically smaller and loaded separately)
        row = self.df.iloc[index]
        data = torch.load(row.method_tensor_path)
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


def get_classification_datasets(
    fold,
    data,
    y_label,
    encoder,
    method,
    tile_index=None,
    endpoint="Overall survival",
    event_label="Deceased",
    n_subsamples=None,
    random_seed=None,
    noise_std=None,
    cache_in_memory=False,
):
    df = pd.read_csv(data)
    columns = [encoder, y_label]
    for col in [tile_index, endpoint, event_label]:
        if col is not None:
            columns.append(col)
    rename = {
        encoder: "encoder",
        y_label: "y",
        tile_index: "tile_index",
        endpoint: "endpoint",
        event_label: "event_label",
    }
    df_train = (
        df.loc[df[f"{fold}_{y_label}"] == "train", columns]
        .reset_index(drop=True)
        .dropna()
        .rename(columns=rename)
    )
    df_val = (
        df.loc[df[f"{fold}_{y_label}"] == "val", columns]
        .reset_index(drop=True)
        .dropna()
        .rename(columns=rename)
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
        # Apply augmentations and caching to training set, only caching to validation
        dset_train = slide_dataset_classification_graph(
            df_train,
            n_subsamples=n_subsamples,
            random_seed=random_seed,
            noise_std=noise_std,
            cache_in_memory=cache_in_memory,
        )
        dset_val = slide_dataset_classification_graph(
            df_val, cache_in_memory=cache_in_memory
        )  # Cache validation but no augmentations
        dset_test = (
            slide_dataset_classification_graph(df_test, cache_in_memory=cache_in_memory)
            if df_test is not None
            else None
        )
    else:
        # Apply augmentations and caching to training set, only caching to validation
        dset_train = slide_dataset_classification(
            df_train,
            n_subsamples=n_subsamples,
            random_seed=random_seed,
            noise_std=noise_std,
            cache_in_memory=cache_in_memory,
        )
        dset_val = slide_dataset_classification(
            df_val, cache_in_memory=cache_in_memory
        )  # Cache validation but no augmentations
        dset_test = (
            slide_dataset_classification(df_test, cache_in_memory=cache_in_memory)
            if df_test is not None
            else None
        )

    return dset_train, dset_val, dset_test


class slide_dataset_classification(torch.utils.data.Dataset):
    """
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    """

    def __init__(
        self,
        df,
        n_subsamples=None,
        random_seed=None,
        noise_std=None,
        cache_in_memory=False,
    ):
        self.df = df
        self.n_subsamples = n_subsamples
        self.random_seed = random_seed
        self.noise_std = noise_std  # Standard deviation for Gaussian noise
        self.cache_in_memory = cache_in_memory
        self.cached_features = {}  # Dictionary to store cached features

        if random_seed is not None:
            np.random.seed(random_seed)

        # Cache all features in memory if requested
        if self.cache_in_memory:
            print(f"Caching {len(self.df)} samples in memory...")
            self._cache_all_features()
            print("Caching completed!")

    def _cache_all_features(self):
        """Load all features into memory"""
        for idx, row in self.df.iterrows():
            path_to_data = row["encoder"]

            # Load the feature data
            data = np.load(path_to_data)
            try:
                feat = data["features"]
            except:
                feat = data

            # Apply tile index if available
            if "tile_index" in row.keys() and not pd.isna(row["tile_index"]):
                tile_index = np.load(row["tile_index"])
                feat = feat[tile_index]

            # Store in cache using the dataframe index as key
            self.cached_features[idx] = feat

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = row["y"]
        time_to_event = row["endpoint"] if "endpoint" in row else None
        censored = ~row["event_label"] if "event_label" in row else None

        # Get features from cache or load from disk
        if self.cache_in_memory:
            # Get the actual dataframe index (not the iloc index)
            df_index = self.df.index[index]
            selected_feat = self.cached_features[
                df_index
            ].copy()  # Copy to avoid modifying cached data
        else:
            # Load from disk (original behavior)
            path_to_data = row["encoder"]
            data = np.load(path_to_data)
            try:
                feat = data["features"]
            except:
                feat = data

            if "tile_index" in row.keys() and not pd.isna(row["tile_index"]):
                tile_index = np.load(row["tile_index"])
                feat = feat[tile_index]
            selected_feat = feat

        # Apply random subsampling if specified
        if self.n_subsamples is not None and len(selected_feat) > 0:
            n_tiles = len(selected_feat)
            if n_tiles <= self.n_subsamples:
                # If we have fewer tiles than requested, use all tiles
                pass  # selected_feat remains unchanged
            else:
                # Randomly sample n_subsamples tiles
                selected_indices = np.random.choice(
                    n_tiles, size=self.n_subsamples, replace=False
                )
                selected_feat = selected_feat[selected_indices]

        # Apply Gaussian noise augmentation if specified
        if self.noise_std is not None and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, selected_feat.shape)
            selected_feat = selected_feat + noise.astype(selected_feat.dtype)

        return {
            "features": selected_feat,
            "target": label,
            "time_to_event": time_to_event,
            "censored": censored,
        }


# TODO: implement this class
class slide_dataset_classification_graph(slide_dataset_classification):
    def __init__(
        self,
        df,
        n_subsamples=None,
        random_seed=None,
        noise_std=None,
        cache_in_memory=False,
    ):
        super(slide_dataset_classification_graph, self).__init__(
            df, n_subsamples, random_seed, noise_std, cache_in_memory
        )

    def __getitem__(self, index):
        # Load data using the parent class method (which includes caching, subsampling and noise)
        item = super(slide_dataset_classification_graph, self).__getitem__(index)

        # Additional graph-specific data extraction (these are typically smaller and loaded separately)
        row = self.df.iloc[index]
        data = torch.load(row.method_tensor_path)
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


def create_balanced_sampler(dataset):
    """
    Create a balanced sampler for handling class imbalance.
    """
    # Get all targets from the dataset
    targets = []
    for i in range(len(dataset)):
        item = dataset[i]
        targets.append(item["target"])

    targets = np.array(targets)

    # Calculate class weights
    classes = np.unique(targets)
    class_weights = compute_class_weight("balanced", classes=classes, y=targets)

    # Create sample weights
    sample_weights = np.zeros(len(targets))
    for i, class_label in enumerate(classes):
        sample_weights[targets == class_label] = class_weights[i]

    # Create WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # print(f"Class distribution: {np.bincount(targets)}")
    # print(f"Class weights: {dict(zip(classes, class_weights))}")

    return sampler


def collate_batch(batch):
    """
    Collation function for stacking batches of variable-length tiles.
    Groups features into a list and stacks other elements into tensors.
    """
    features = [torch.Tensor(item["features"]) for item in batch]
    targets = [item["target"] for item in batch]
    time_to_events = (
        [item["time_to_event"] for item in batch]
        if "time_to_event" in batch[0] and batch[0]["time_to_event"] is not None
        else None
    )
    censored = (
        [item["censored"] for item in batch]
        if "censored" in batch[0] and batch[0]["censored"] is not None
        else None
    )

    return {
        "features": features,
        "target": targets,
        "time_to_event": time_to_events,
        "censored": censored,
    }
