import torch
from torch.nn import BCELoss


class TripletLoss:
    def __init__(self, margin):
        if not isinstance(margin, float):
            raise TypeError("margin should be a float")
        self._margin = margin

    def __call__(self, y_pred, y_true):
        z0 = torch.zeros_like(y_pred)
        pos_avg = torch.sum(torch.where(y_true == 1, y_pred, z0), dim=-1)
        # avoid nan
        denom1 = torch.sum(y_true, dim=-1)
        pos_avg /= torch.max(torch.ones_like(denom1), denom1)
        neg_avg = torch.sum(torch.where(y_true == 0, y_pred, z0), dim=-1)
        # avoid nan
        denom2 = torch.sum(1 - y_true, dim=-1)
        neg_avg /= torch.max(torch.ones_like(denom2), denom2)
        loss = torch.max(torch.zeros_like(pos_avg),
                         neg_avg - pos_avg + self._margin)
        # loss = max(0, pos_dist - neg_dist + margin)
        #      = max(0, neg_sims - pos_sims + margin
        return torch.mean(loss)


class CrossEntropyLoss:
    def __init__(self):
        self._func = BCELoss()

    def __call__(self, y_pred, y_true):
        return self._func(y_pred, y_true.float())
