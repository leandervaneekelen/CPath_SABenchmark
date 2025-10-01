import os
import numpy as np
import torch

# Import aggregation methods
from aggregator import (
    GMA,
    LoopABMIL,
    TransMIL,
    DSMIL,
    CLAM_SB,
    CLAM_MB,
    VarAttention,
    GTP,
    PatchGCN_Surv,
    DeepGraphConv_Surv,
    MIL_Sum_FC_surv,
    MIL_Attention_FC_surv,
    MIL_Cluster_FC_surv,
    PureTransformer,
)

AGGREGATION_METHODS = [
    "AB-MIL",
    "AB-MIL_FC_small",
    "AB-MIL_FC_big",
    "LoopABMIL",
    "VarMIL",
    "CLAM_SB",
    "CLAM_MB",
    "ViT_MIL",
    "transMIL",
    "DS-MIL",
    "GTP",
    "PatchGCN",
    "DeepGraphConv",
]


def get_aggregator(method, n_classes, ndim=1024, **kwargs):
    # GMA
    if method == "AB-MIL":
        return GMA(n_classes=n_classes, ndim=ndim, **kwargs)
    elif method == "AB-MIL_FC_small":
        return MIL_Attention_FC_surv(
            ndim=ndim, n_classes=n_classes, size_arg="small", **kwargs
        )
    elif method == "AB-MIL_FC_big":
        return MIL_Attention_FC_surv(
            ndim=ndim, n_classes=n_classes, size_arg="big", **kwargs
        )
    elif method == "LoopABMIL":
        return LoopABMIL(
            input_dim=ndim, hidden_dim=512, output_dim=n_classes, **kwargs
        )  # TODO: hidden dim may be chosen better
    # DeepSMILE
    elif method == "VarMIL":
        return VarAttention(n_classes=n_classes, ndim=ndim, **kwargs)
    # CLAM
    if method == "CLAM_SB":
        return CLAM_SB(n_classes=n_classes, ndim=ndim, **kwargs)
    elif method == "CLAM_MB":
        return CLAM_MB(n_classes=n_classes, ndim=ndim, **kwargs)
    # TransMIL
    elif method == "ViT_MIL":
        return PureTransformer(n_classes=n_classes, ndim=ndim, **kwargs)
    elif method == "transMIL":
        return TransMIL(n_classes=n_classes, ndim=ndim, **kwargs)
    # DSMIL
    elif method == "DS-MIL":
        return DSMIL(n_classes=n_classes, ndim=ndim, **kwargs)
    # GTP
    elif method == "GTP":
        return GTP(n_classes=n_classes, ndim=ndim, **kwargs)
    # PatchGCN
    elif method == "PatchGCN":
        return PatchGCN_Surv(n_classes=n_classes, ndim=ndim, **kwargs)
    # DeepGraphConv
    elif method == "DeepGraphConv":
        return DeepGraphConv_Surv(n_classes=n_classes, ndim=ndim, **kwargs)
    # elif method == 'MIL_Sum_FC':
    #     return MIL_Sum_FC_surv(ndim=ndim, n_classes=2, **kwargs)
    # elif method == 'MIL_Cluster_FC':
    #     return MIL_Cluster_FC_surv(ndim=ndim, n_classes=2, **kwargs)
    # elif method == 'DTMIL':
    #     return DTMIL(ndim=ndim, n_classes=2, **kwargs)
    else:
        raise Exception(f"Method {method} not defined")


def model_forward_pass(batch, model, method, device):
    """
    Handles the forward pass for all model flavors based on `method`.
    Returns the results_dict from the model.
    """

    # Get features, reshape and copy to GPU
    if method in ["ViT_MIL", "DTMIL"]:
        feat = batch["feat_map"].float().permute(0, 3, 1, 2)
        feat = feat.to(device)
    else:
        feat = batch["features"]
        if isinstance(feat, list):
            # Handle batch case - features is a list of tensors
            feat = [f.to(device) for f in feat]
        else:
            # Handle single sample case
            feat = feat.to(device)

    if method in ["GTP"]:
        adj = batch["adj_mtx"].float().to(device)
        mask = batch["mask"].float().to(device)
        results_dict = model(feat, adj, mask)
    elif method in ["PatchGCN", "DeepGraphConv"]:
        edge_index = batch["edge_index"].squeeze(0).to(device)
        edge_latent = batch["edge_latent"].squeeze(0).to(device)
        results_dict = model(feat=feat, edge_index=edge_index, edge_latent=edge_latent)
    # elif args.method in ['DTMIL']:
    #     mask = input['mask'].bool().to(device)
    #     tensors = NestedTensor(feat, mask)
    #     results_dict = model(tensors)
    else:
        results_dict = model(feat)
    return results_dict