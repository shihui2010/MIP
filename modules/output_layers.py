import torch
import torch.nn as nn


def _sims(user_emb, item_emb):
    """
    n_embs is the number of embedding vecters per user
    :param user_emb: tensor, shape [batch_size, n_embs, emb_size]
    :param item_emb: tensor, shape [batch_size, n_items, emb_size]
    :return: similarity prediction [batch_size, n_items, n_embs]
    """
    # print(torch.norm(user_emb, dim=-1).shape)
    user_emb_norm = user_emb / torch.clamp(
        torch.norm(user_emb, dim=-1, keepdim=True), 1e-7)
    item_emb_norm = item_emb / torch.clamp(
        torch.norm(item_emb, dim=-1, keepdim=True), 1e-7)
    # print(user_emb_norm.shape, item_emb_norm.shape)
    user_4d = torch.unsqueeze(user_emb_norm, dim=1)
    item_4d = torch.unsqueeze(item_emb_norm, dim=2)
    cos_sim = torch.sum(torch.mul(user_4d, item_4d), dim=-1, keepdim=True)
    # print(cos_sim.shape)
    return cos_sim


class SimilarityOutputLayer(nn.Module):
    def __init__(self):
        super(SimilarityOutputLayer, self).__init__()
        self._projection_layer = nn.Sequential(
            nn.Linear(1, 1, True), nn.Sigmoid())

    def forward(self, user_emb, item_emb):
        """
        n_embs is the number of embedding vecters per user
        :param user_emb: tensor, shape [batch_size, n_embs, emb_size]
        :param item_emb: tensor, shape [batch_size, n_items, emb_size]
        :return: similarity prediction [batch_size, n_items, n_embs]
        """
        # print(torch.norm(user_emb, dim=-1).shape)
        cos_sim = _sims(user_emb, item_emb)
        return torch.squeeze(self._projection_layer(cos_sim), dim=-1)


class MaxAggregator(nn.Module):
    def __init__(self):
        super(MaxAggregator, self).__init__()
        self._output_layer = SimilarityOutputLayer()

    def forward(self, user_emb, item_emb, last_item_mask=None):
        sims = self._output_layer(user_emb, item_emb)
        # print("max aggregator", sims.shape, user_emb.shape, item_emb.shape)
        if last_item_mask is None:
            return sims.max(dim=-1, keepdim=False)[0]
        last_item_mask = last_item_mask.unsqueeze(dim=1)
        sims *= torch.where(last_item_mask == 1,
                            last_item_mask,
                            torch.ones_like(last_item_mask) * -10000)
        return sims.max(dim=-1, keepdim=False)[0]


class WeightedMaxAggregator(nn.Module):
    def __init__(self):
        super(WeightedMaxAggregator, self).__init__()
        self._projection_layer = nn.Sequential(
            nn.Linear(1, 1, True), nn.Sigmoid())

    def forward(self, user_emb, item_emb, cluster_weights, last_item_mask):
        """
        :param user_emb: [batch_size, seq_len, emb_size]
        :param item_emb: [batch_size, n_items, emb_size]
        :param cluster_weights: [batch_size, seq_len, 1]
        :param last_item_mask: [batch_size, 1, seq_len]
        :return:
        """
        # print("cluster weight", cluster_weights.shape)
        sims = _sims(user_emb, item_emb)
        # print("sims shape", sims.shape, 'cluster weights', cluster_weights.shape)
        sims = sims * cluster_weights.unsqueeze(dim=2)
        # print("sims shape", sims.shape)
        # print("last item mask shape", last_item_mask.shape)
        #last_item_mask = last_item_mask.unsqueeze(dim=1)
        sims = self._projection_layer(sims).squeeze(dim=-1)
        # print("sims shape", sims.shape, "last item shape", last_item_mask.shape)
        sims = sims.permute(0, 2, 1) * last_item_mask
        sims = sims.max(dim=2, keepdim=False)[0] 
        # print("sims final shape", sims.shape)
        return sims
        # return self._projection_layer(
        #     sims.max(dim=-1, keepdim=False)[0])
