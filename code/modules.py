import os
import numpy as np
import torch

# Import aggregation methods
from aggregator import (
    GMA,
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

# PureTransformer, DTMIL


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
