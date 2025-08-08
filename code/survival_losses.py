import torch
import torch.nn as nn
from pycox.models.loss import CoxPHLoss
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LossFactory:
    def __init__(self, loss, **kwargs):
        if loss == "nll":
            self.loss = NLLSurvLoss(**kwargs)
        elif loss == "cox":
            self.loss = CoxSurvLoss(**kwargs)
        else:
            raise NotImplementedError(
                f"Loss {loss} not implemented. Available losses: nll, cox"
            )

    def get_loss(self):
        return self.loss


def nll_loss(hazards, survival, Y, c, alpha=0.4, eps=1e-7):
    """
    Continuous time scale divided into k discrete bins: T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
    Y = T_discrete is the discrete event time:
        - Y = -1 if T_cont \in (-inf, 0)
        - Y = 0 if T_cont \in [0, a_1)
        - Y = 1 if T_cont in [a_1, a_2)
        - ...
        - Y = k-1 if T_cont in [a_(k-1), inf)
    hazards = discrete hazards, hazards(t) = P(Y=t | Y>=t, X) for t = -1, 0, 1, 2, ..., k-1
    survival = survival function, survival(t) = P(Y > t | X)

    All patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
    -> hazards(-1) = 0
    -> survival(-1) = P(Y > -1 | X) = 1

    Summary:
        - neural network is hazard probability function, h(t) for t = 0, 1, 2, ..., k-1
        - h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 0, 1, 2, ..., k-1
    c = c.view(batch_size, 1).float()  # censoring status, 0 or 1
    if survival is None:
        survival = torch.cumprod(
            1 - hazards, dim=1
        )  # survival is cumulative product of 1 - hazards
    survival_padded = torch.cat(
        [torch.ones_like(c), survival], 1
    )  # survival(-1) = 1, all patients are alive from (-inf, 0) by definition
    # after padding, survival(t=-1) = survival[0], survival(t=0) = survival[1], survival(t=1) = survival[2], etc
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(survival_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(
        torch.gather(survival_padded, 1, Y + 1).clamp(min=eps)
    )
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, survival, Y, c, alpha=None, **kwargs):
        if alpha is None:
            return nll_loss(hazards, survival, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, survival, Y, c, alpha=alpha)


class CoxSurvLoss(object):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, hazards, survival, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(survival)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = survival[j] >= survival[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean(
            (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c.int())
        )
        return loss_cox


class PycoxCoxphLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        event = targets[:, 0]
        duration = targets[:, 1]
        if len(targets) == 1:
            raise ValueError("for coxph batch size has to be > 1")

        if event.max() > 1:
            raise ValueError(
                "expects events max to be 1, not events.max(), probably confused with durations"
            )
        loss = self.loss(logits, duration, event)
        return loss
