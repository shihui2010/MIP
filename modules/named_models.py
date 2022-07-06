from typing import Union
from modules.attention import MHGlobalQueryAttention, MultiHeadAttention
from modules.loss_funcs import CrossEntropyLoss, TripletLoss
from modules.output_layers import MaxAggregator, WeightedMaxAggregator
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
from modules.attention import OneHot, RelValue, SineEnc


def _get_cluster_m(n_clusters, method):
    if method == 'ward':
        return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif method == 'dbscan':
        return cluster.DBSCAN(eps=.2)
    elif method == 'spectrum':
        return cluster.SpectralClustering(
            n_clusters=n_clusters,
            eigen_solver='arpack',
            affinity="nearest_neighbors")
    elif method == 'birch':
        return cluster.Birch(n_clusters=n_clusters)
    elif method == 'kmeans':
        return cluster.KMeans(n_clusters=n_clusters, random_state=0)
    else:
        raise NotImplementedError(f"Cluster method {method}")


class _AttentionModel(nn.Module):
    def __init__(self, hps, attention_cls):
        """
        :param hps: dict
        :param attention_cls: [MHGlobalQueryAttention,
                               MultiHeadAttention,
                                ComiRecAttention]
        """
        super(_AttentionModel, self).__init__()
        if hps['activation'] == 'relu':
            act_fn = nn.ReLU
        elif hps["activation"] == 'sigmoid':
            act_fn = nn.Sigmoid
        elif hps["activation"] == 'tanh':
            act_fn = nn.Tanh
        else:
            act_fn = nn.Identity

        if "item_count" in hps:
            self._item_embedding = nn.Embedding(
                num_embeddings=hps["item_count"],
                embedding_dim=hps["emb_dim"])
        else:
            self._item_embedding = None

        hps["input_dim"] = hps["emb_dim"]
        if hps["att_on_tem"]:
            if hps["tem_enc"] is None:
                self._tem_enc_layer = RelValue()
            elif hps["tem_enc"] == "one_hot":
                self._tem_enc_layer = OneHot(max_k=hps["tem_dim"] - 1)
            elif hps["tem_enc"] == "sine":
                from modules.attention import SineEnc
                self._tem_enc_layer = SineEnc(dim=hps["tem_dim"])
            else:
                raise AttributeError("unknown encoding layer {hps['tem_enc']}")
            hps["input_dim"] += self._tem_enc_layer.output_dim
        else:
            self._tem_enc_layer = None

        if hps["att_on_pos"]:
            from modules.attention import SineEnc
            if attention_cls == TisasAttention:
                hps["pos_dim"] = hps["emb_dim"]
            else:
                hps["input_dim"] += hps["pos_dim"]
            self._pos_enc = SineEnc(dim=hps["pos_dim"])
        else:
            self._pos_enc = None

        self._attention = attention_cls(
            d_model=hps["d_model"],
            emb_size=hps["emb_dim"],
            input_dim=hps["input_dim"],
            n_head=hps["n_head"],
            dropout=hps["dropout"],
            activation=act_fn,
            share_query=hps["share_query"],
            seq_len=hps["seq_len"]
        )
        self._sims = MaxAggregator()

        if hps["loss_fn"] == "triplet":
            self._loss_fn = TripletLoss(hps["loss_margin"])
        else:
            self._loss_fn = CrossEntropyLoss()

    def _add_optim(self, hps):
        self._optimizer = Adam(self.parameters(), lr=hps['lr'])
        self._hps = hps
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            self.to(self._device)
        else:
            self._device = torch.device('cpu')

    def forward(self, seq, pos, neg, timestamps=None):
        pos = self._embed_item(pos)
        neg = self._embed_item(neg)
        if torch.any(torch.isnan(pos)):
            print("pos embedding has nan", pos)
            print("pos sequence", pos)
        if torch.any(torch.isnan(neg)):
            print("neg embeddings has nan", neg)
            print("neg sequence", neg)
        # print("forward got timestmaps", timestamps.shape if timestamps is not None else None)
        user_embs = self._embed_user(seq, timestamps)
        if torch.any(torch.isnan(user_embs)):
            print("user embeddings has nan", user_embs)

        # print("user_embedding shape", user_embs.shape)
        pos_sims = self._sims(user_embs, pos)
        neg_sims = self._sims(user_embs, neg)
        if torch.any(torch.isnan(pos_sims)):
            print('pos sims has nan', pos_sims)
        if torch.any(torch.isnan(neg_sims)):
            print('neg sims has nan', neg_sims)
        return pos_sims, neg_sims

    def _embed_user(self, seq, timestamps=None):
        if timestamps is not None and self._tem_enc_layer is not None:
            time_enc = self._tem_enc_layer(timestamps)
        else:
            time_enc = None
        if self._pos_enc is not None:
            positions = torch.arange(0, seq.shape[1]).to(self._device)
            positions = torch.reshape(positions, [1, -1, 1])
            pos_enc = self._pos_enc(positions.repeat(seq.shape[0], 1, 1))
        else:
            pos_enc = None
        seq_emb = self._embed_item(seq)
        user_emb = self._attention(seq_emb, time_enc=time_enc,
                                   pos_enc=pos_enc, causality=False)
        return user_emb

    def _embed_item(self, item):
        if self._item_embedding is None:
            return item.float()
        return self._item_embedding(item.long())

    def _get_loss(self, pos_logits, neg_logits):
        truth_1 = torch.ones_like(pos_logits)
        truth_0 = torch.zeros_like(neg_logits)
        y_truth = torch.cat([truth_1, truth_0], dim=1)
        y_pred = torch.cat([pos_logits, neg_logits], dim=1)
        #print('y_truth shape', y_truth.shape, "y_pred shape", y_pred.shape)
        return self._loss_fn(y_pred, y_truth)

    def supervised(self, seq, timestamps, pos, neg, train=True):
        """
        :param seq:
        :param timestamps:
        :param pos_enc: position encoding
        :param pos:
        :param neg:
        :param train:
        :return:
        """
        pos_logits, neg_logits = self.forward(seq, pos, neg, timestamps)
        loss = self._get_loss(pos_logits, neg_logits)
        if train:
            self._optimizer.zero_grad()
            loss.backward()
            res = [d.grad for d in self.parameters() if
                   d.grad is not None and torch.any(torch.isnan(d.grad))]
            if len(res):
                print("Nan grad encountered", res)
                return pos_logits, neg_logits, loss
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            self._optimizer.step()
        return pos_logits, neg_logits, loss

    def clustered_inference(self, seq, timestamps, pos, neg,
                            pos_enc=None,
                            n_clusters=10,
                            selection: str ='last',
                            method='ward'):
        with torch.no_grad():
            user_emb = self._embed_user(seq, timestamps)
            user_emb_np = user_emb.cpu().numpy()
            centers = np.empty([user_emb_np.shape[0], 
                n_clusters, user_emb_np.shape[2]])
            for bid in range(user_emb_np.shape[0]):
                cluster_m = _get_cluster_m(n_clusters, method)
                c_labels = cluster_m.fit_predict(user_emb_np[bid])
                distinct = np.unique(c_labels)
                # print("before cluster shape", user_emb_np.shape)
                # print("center shape", centers.shape)
                for cid, c in enumerate(distinct):
                    cluster = np.take(
                        user_emb_np[bid], np.where(c_labels == c)[0], axis=0)
                    if cluster.shape[0] <= 2:
                        # when cluster contains two nodes, there's no medoid
                        centers[bid][cid] = cluster[-1]
                        continue
                    # print("cluster shape", cluster.shape)
                    if selection == 'last':
                        centers[bid][cid] = cluster[-1]
                    else:
                        pairwise = squareform(pdist(cluster))
                        # print("pairwise", pairwise.shape)
                        row_sum = np.sum(pairwise, axis=0)
                        # print("row sum", row_sum)
                        idx = np.argmin(row_sum)
                        # print("medoid id", idx)
                        centers[bid][cid] = cluster[idx]

                # for those cluster methods that return
                # arbitrary number of clusters
                for cid in range(len(distinct), n_clusters):
                    centers[bid][cid] = centers[bid][cid % len(distinct)]
            #print("clustered user emb", user_emb.shape, centers.shape)
            #print(centers)
            user_emb = torch.from_numpy(centers).float().to(self._device)
            pos_emb = self._embed_item(pos)
            neg_emb = self._embed_item(neg)
            pos_sims = self._sims(user_emb, pos_emb)
            neg_sims = self._sims(user_emb, neg_emb)
            #print(pos_sims, neg_sims)
            loss = self._get_loss(pos_sims, neg_sims)
        return pos_sims, neg_sims, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GlobalQueryModel(_AttentionModel):
    def __init__(self, hps):
        super(GlobalQueryModel, self).__init__(hps, MHGlobalQueryAttention)
        self._add_optim(hps)

class StaticPinnerSagePlus(_AttentionModel):
    def __init__(self, hps):
        super(StaticPinnerSagePlus, self).__init__(hps, MultiHeadAttention)
        self._add_optim(hps)


class WeightedPinnerSagePlus(_AttentionModel):
    def __init__(self, hps):
        hps["att_on_tem"] = True
        hps["tem_enc"] = 'sine'
        super(WeightedPinnerSagePlus, self).__init__(hps, MultiHeadAttention)
        # overwrite
        self._sims = WeightedMaxAggregator()
        self._weight_tem_enc = OneHot(max_k=hps["tem_dim"])
        input_dim = hps["seq_len"] * self._weight_tem_enc.output_dim + hps["emb_dim"]
        # print('weight input dim', input_dim)
        self._weight_model = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=1),
            nn.Softplus()
            )
        self._add_optim(hps)

    def load_unweighted(self, ckpt):
        state_dict = torch.load(ckpt).state_dict()
        own_state = self.state_dict()
        print("The following parameters are loaded from {ckpt}")
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
                print(name)
            except Exception as e:
                print(f"failed to load {name} due to:\n {e}")
        print("\n")

    def _cluster_items(self, seq_embed, n_clusters=5, method='ward'):
        cluster_mask = torch.zeros(
            [seq_embed.shape[0], seq_embed.shape[1], seq_embed.shape[1]])
        last_item_mask = torch.zeros(
            [seq_embed.shape[0], 1, seq_embed.shape[1]])
        for bid in range(seq_embed.shape[0]):
            cluster_m = _get_cluster_m(n_clusters, method)
            c_labels = cluster_m.fit_predict(seq_embed[bid].detach().cpu())
            distinct = np.unique(c_labels)
            for cid, c in enumerate(distinct):
                item_idx = np.where(c_labels == c)[0]
                for i in item_idx:
                    for j in item_idx:
                        cluster_mask[bid, i, j] = 1
                last_item_mask[bid, 0, item_idx[-1]] = 1
        return cluster_mask.to(seq_embed.device), last_item_mask.to(seq_embed.device)

    def _compute_weights(self, cluster_emb, timestamps, cluster_mask):
        itvl = timestamps.squeeze(dim=-1).unsqueeze(dim=1) - timestamps
        # print("itvl shape", itvl.shape)
        itvl_enc = self._weight_tem_enc(time_itvls=torch.abs(itvl))
        # print("itvl enc", itvl_enc.shape)
        # [batch_size, seq_len, seq_len, tem_dim]
        itvl_enc = itvl_enc * cluster_mask.unsqueeze(dim=-1)
        itvl_enc = itvl_enc.reshape([itvl.shape[0], itvl.shape[1], -1])
        # print("itvl enc reshape", itvl_enc.shape)
        # [batch_size, seq_len, seq_len * tem_dim]
        x = torch.cat([itvl_enc, cluster_emb], dim=-1)
        # print("input x shape", x.shape)
        cluster_weights = self._weight_model(x)
        # print("cluster weights", cluster_weights.shape)
        # print("cluster weights", cluster_weights)
        return cluster_weights

    def forward(self, seq, pos, neg, timestamps, n_clusters=5, method='ward'):
        assert timestamps is not None, "Weighted model requires timestamps"
        pos = self._embed_item(pos)
        neg = self._embed_item(neg)
        if torch.any(torch.isnan(pos)):
            print("pos embedding has nan", pos)
            print("pos sequence", pos)
        if torch.any(torch.isnan(neg)):
            print("neg embeddings has nan", neg)
            print("neg sequence", neg)

        # embed user
        time_enc = self._tem_enc_layer(timestamps)
        if self._pos_enc is not None:
            positions = torch.arange(0, seq.shape[1]).to(self._device)
            positions = torch.reshape(positions, [1, -1, 1])
            pos_enc = self._pos_enc(positions.repeat(seq.shape[0], 1, 1))
        else:
            pos_enc = None
        seq_emb = self._embed_item(seq)
        cluster_mask, last_item_mask = self._cluster_items(
            seq_emb, n_clusters=n_clusters, method=method)
        user_embs = self._attention(seq_emb, time_enc=time_enc,
                                   pos_enc=pos_enc, causality=False,
                                   cluster_mask=cluster_mask)

        if torch.any(torch.isnan(user_embs)):
            for bid in range(user_embs.shape[0]):
                if not torch.any(torch.isnan(user_embs[bid])):
                    continue
                print("user embeddings has nan", user_embs[bid])
                print("cluster mask", cluster_mask[bid])

        cluster_weight = self._compute_weights(
                user_embs, timestamps, cluster_mask)
        # [batch_size, seq_len, 1]
        self.last_embs = user_embs
        self.last_weight = cluster_weight
        self.last_cluster = cluster_mask

        #print("user_embedding shape", user_embs.shape)
        pos_sims = self._sims(user_embs, pos, cluster_weight, last_item_mask)
        neg_sims = self._sims(user_embs, neg, cluster_weight, last_item_mask)
        # print("pos sims", pos_sims)
        # print("neg sims", neg_sims)
        if torch.any(torch.isnan(pos_sims)):
            print('pos sims has nan', pos_sims)
        if torch.any(torch.isnan(neg_sims)):
            print('neg sims has nan', neg_sims)
        return pos_sims, neg_sims

    def supervised(self, seq, timestamps, pos, neg, train=True,
                   n_clusters=5, method='ward'):
        pos_logits, neg_logits = self.forward(
            seq, pos, neg, timestamps, n_clusters=n_clusters, method=method)
        loss = self._get_loss(pos_logits, neg_logits)
        if train:
            self._optimizer.zero_grad()
            loss.backward()
            res = [d.grad for d in self.parameters() if
                   d.grad is not None and torch.any(torch.isnan(d.grad))]
            if len(res):
                print("Nan grad encountered", res)
                return pos_logits, neg_logits, loss
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            self._optimizer.step()
        return pos_logits, neg_logits, loss

    def clustered_inference(self, seq, timestamps, pos, neg,
                            pos_enc=None,
                            n_clusters=10,
                            selection: str ='last',
                            method='ward'):
        return self.supervised(seq, timestamps, pos, neg,
                               train=False, n_clusters=n_clusters,
                               method=method)


class ExpDecayWeightModel(WeightedPinnerSagePlus):
    def __init__(self, hps):
        super(ExpDecayWeightModel, self).__init__(hps)
        self._sims = WeightedMaxAggregator()
        self._lambda = hps['lambda']

    def _compute_weights(
            self, cluster_emb, timestamps, cluster_mask):
        timestamps *= 60 * 60 * 24  # back to seconds
        time_itvls = timestamps[:, -1:, :] - timestamps
        time_itvls *= -self._lambda
        exponentials = torch.exp(time_itvls).repeat(1, 1, cluster_mask.shape[-1])
        weights = torch.sum(exponentials * cluster_mask, dim=-1, keepdims=True)
        return weights



