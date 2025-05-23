import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchgcn_utils.model_utils import *

################################
### Deep Sets Implementation ###
################################
class MIL_Sum_FC_surv(nn.Module):
    def __init__(self, size_arg="small", dropout=0.25, ndim=1024, n_classes=4):
        r"""
        Deep Sets Implementation.

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Sum_FC_surv, self).__init__()
        self.size_dict_path = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        self.phi = nn.Sequential(
            *[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        )
        self.rho = nn.Sequential(
            *[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)]
        )

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.phi = nn.DataParallel(self.phi, device_ids=device_ids).to("cuda:0")

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs["feat"]

        h_path = self.phi(x_path).sum(axis=0)
        h_path = self.rho(h_path)
        h = h_path  # [256] vector

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat}
        return results_dict


################################
# Attention MIL Implementation #
################################
class MIL_Attention_FC_surv(nn.Module):
    def __init__(self, size_arg="small", dropout=0.25, ndim=1024, n_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Attention_FC_surv, self).__init__()
        self.size_dict_path = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(
            L=size[1], D=size[2], dropout=dropout, n_classes=1
        )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(
            *[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)]
        )

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(
                self.attention_net, device_ids=device_ids
            ).to("cuda:0")

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, feat):
        x_path = feat

        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        h = h_path  # [256] vector

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat}
        return results_dict


######################################
# Deep Attention MISL Implementation #
######################################
class MIL_Cluster_FC_surv(nn.Module):
    def __init__(
        self, num_clusters=10, size_arg="small", dropout=0.25, ndim=1024, n_classes=4
    ):
        r"""
        Attention MIL Implementation

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Cluster_FC_surv, self).__init__()
        self.size_dict_path = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}
        self.num_clusters = num_clusters

        ### FC Cluster layers + Pooling
        size = self.size_dict_path[size_arg]
        phis = []
        for phenotype_i in range(num_clusters):
            phi = [
                nn.Linear(size[0], size[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(size[1], size[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            phis.append(nn.Sequential(*phi))
        self.phis = nn.ModuleList(phis)
        self.pool1d = nn.AdaptiveAvgPool1d(1)

        ### WSI Attention MIL Construction
        fc = [nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(
            L=size[1], D=size[2], dropout=dropout, n_classes=1
        )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(
            *[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)]
        )

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(
                self.attention_net, device_ids=device_ids
            ).to("cuda:0")
        else:
            self.attention_net = self.attention_net.to(device)

        self.phis = self.phis.to(device)
        self.pool1d = self.pool1d.to(device)
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs["feat"]
        cluster_id = kwargs["centroid"].detach().cpu().numpy()

        print("cluster_id:", cluster_id.shape)

        ### FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            h_cluster_i = self.phis[i](x_path[cluster_id == i])
            if h_cluster_i.shape[0] == 0:
                h_cluster_i = torch.zeros((1, 512)).to(torch.device("cuda"))
            h_cluster.append(self.pool1d(h_cluster_i.T.unsqueeze(0)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0)

        ### Attention MIL
        A, h_path = self.attention_net(h_cluster)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        h = h_path

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat}
        return results_dict
