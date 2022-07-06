import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Union


def _gen_timing_signal(length, channels,
                       min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/
    layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(max_timescale) /
                                      float(min_timescale)) /
                               (float(num_timescales) - 1))
    inv_timescales = min_timescale * \
                     np.exp(np.arange(num_timescales).astype(np.float) *
                            -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * \
                  np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _timestamp_to_intervals(timestamps: torch.Tensor) -> torch.Tensor:
    """
    convert timestamps to intervals via numpy since
    tf tensor does not support assignment
    :param timestamps: python list or np array
    :return: np array
    """
    with torch.no_grad():
        itvl = torch.zeros_like(timestamps)
        itvl[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
    return itvl


def _log(x: torch.Tensor, base: int) -> torch.Tensor:
    """arbitrary base logarithm"""
    if base <= 1.0:
        raise ValueError("base should always larger than 1")
    numerator = torch.log(x)
    denominator = torch.log(torch.ones_like(x) * base)
    res = numerator / denominator
    res = torch.where(torch.isnan(res), torch.zeros_like(res), res)
    return res


def masked_softmax(vec, mask, dim=-1, epsilon=1e-3):
    # trying another implementation
    vec = torch.clamp(vec, max=10)
    vec = vec.masked_fill(mask=torch.logical_not(mask.bool()), value=-np.inf)
    res = F.softmax(vec, dim=dim)
    if torch.any(torch.isnan(res)):
        print("masked softmax created nan")

    return res


class OneHot(nn.Module):
    def __init__(self, base: int = 2, max_k: int = 10):
        super(OneHot, self).__init__()
        self._base = base
        self._max_k = max_k
        self.output_dim = self._max_k + 1

    def forward(self, timestamps: torch.Tensor = None,
                time_itvls=None) -> torch.Tensor:
        with torch.no_grad():
            if time_itvls is None:
                itvls = _timestamp_to_intervals(timestamps)
            else:
                itvls = time_itvls
            log_itvls = _log(itvls, self._base)
            bucket_ids = torch.floor(log_itvls).clamp_(0, self._max_k).long()
            return F.one_hot(torch.squeeze(bucket_ids, dim=2),
                             num_classes=(self._max_k + 1)).float()


class RelValue(nn.Module):
    output_dim = 1

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            itvl = _timestamp_to_intervals(timestamps).float()
            itvl /= torch.mean(itvl)
            return itvl


class SineEnc(nn.Module):
    def __init__(self, dim, max_time_scale=1e4):
        super(SineEnc, self).__init__()
        self._dim = dim
        self._denom = torch.arange(0, self._dim, 2) / self._dim
        self._denom = self._denom.float()
        self._denom *= - torch.log(torch.tensor(max_time_scale, dtype=torch.float))
        self._denom = torch.reshape(torch.exp(self._denom), [1, 1, -1])
        self.output_dim = dim

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        device = timestamps.device
        if self._denom.device != device:
            self._denom = self._denom.to(device)
        #if not isinstance(timestamps, np.ndarray) and not isinstance(timestamps, torch.Tensor):
        #    timestamps = np.expand_dims(np.array(timestamps), axis=-1)
        pe = torch.zeros([timestamps.shape[0], timestamps.shape[1],
                       self._dim], dtype=torch.float).to(device)
        pe[:, :, 0::2] = torch.sin(timestamps * self._denom)
        pe[:, :, 1::2] = torch.cos(timestamps * self._denom)
        return pe


class SingleAttention(nn.Module):
    def __init__(self, d_model, input_dim,
                 activation: nn.Module =nn.Identity):
        super(SingleAttention, self).__init__()
        self._dk = d_model
        self._sqrt_dk = torch.sqrt(torch.tensor(d_model).float())
        self._q_proj = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=self._dk,
                      bias=False),
            activation())
        self._k_proj = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=self._dk,
                      bias=False),
            activation())

    def forward(self, seq, pos_enc, time_enc, causality, cluster_mask=None,
                return_att=False):
        """
        :param seq: [batch_size, seq_len, feature_dim]
        :param pos_enc: [batch_size, seq_len, enc_dim]
        :param time_enc: [batch_size, seq_len, enc_dim]
        :param causality: attention refers only to past items
        :param cluster_mask:
        :return: [batch_size, seq_len, feature_dim]
        """
        q_seq = seq
        k_seq = seq
        device = seq.device
        if pos_enc is not None:
            q_seq = torch.cat([q_seq, pos_enc], dim=-1)
            k_seq = torch.cat([k_seq, pos_enc], dim=-1)
        if time_enc is not None:
            q_seq = torch.cat([q_seq, time_enc], dim=-1)
            k_seq = torch.cat([k_seq, time_enc], dim=-1)
        q = self._q_proj(q_seq)
        k = self._k_proj(k_seq)
        attentions = torch.matmul(q, torch.transpose(k, 1, 2)) / self._sqrt_dk

        # masking
        mask = torch.ones_like(attentions).to(device)
        if causality:
            mask = torch.triu(mask).to(device)
        if cluster_mask is not None:
            mask = mask * cluster_mask
        attentions = masked_softmax(attentions, mask, dim=-1)
        if return_att:
            return torch.matmul(attentions, seq), attentions
        res = torch.matmul(attentions, seq)
        return res


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, input_dim,
                 emb_size, n_head,
                 dropout=0.1, activation: nn.Module =nn.Identity, **unused):
        """
        :param d_model: dimension in attention projection
        :param input_dim: input dimension, including the pos enc and tem enc
        :param emb_size: raw dimension of item feature
        :param n_head:
        :param dropout:
        :param activation:
        :param unused:
        """
        super(MultiHeadAttention, self).__init__()
        self.pos_ff = nn.Sequential(
            nn.Linear(in_features=n_head * emb_size, out_features=emb_size),
            nn.Tanh(),
            nn.Linear(in_features=emb_size, out_features=emb_size),
        )
        self.attention_heads = nn.ModuleList(
            [SingleAttention(d_model, input_dim, activation)
             for _ in range(n_head)])
        self._dropouts = nn.Dropout(dropout)

    def forward(self, seq, pos_enc, time_enc, causality, cluster_mask=None):
        multihead_res = [att(seq, pos_enc, time_enc, causality, cluster_mask)
                         for att in self.attention_heads]
        concats = torch.cat(multihead_res, dim=-1)
        # encoded = self.pos_ff(self._dropouts(concats))
        encoded = self.pos_ff(concats)
        return encoded


class GlobalQueryAttention(nn.Module):
    def __init__(self, d_model, input_dim,
                 dropout=0.1,
                 activation=nn.Identity,
                 bias=True,
                 global_query: Union[None, torch.Tensor] = None):
        super(GlobalQueryAttention, self).__init__()
        self._dk = d_model
        self._k_proj = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=self._dk, bias=bias),
            activation())
        self._dropouts = nn.Dropout(dropout)

        if global_query is None:
            self._query = Parameter(torch.randn([self._dk]), requires_grad=True)
        else:
            assert len(global_query.shape) == 1 and \
                   global_query.shape[0] == d_model, "Invalid query shape"
            if isinstance(global_query, Parameter):
                self._query = global_query
            else:
                self._query = Parameter(global_query, requires_grad=True)

    def forward(self, seq, pos_enc, time_enc, causality, return_att=False):
        """
        :param seq: [batch_size, seq_len, feature_dim]
        :param pos_enc: [batch_size, seq_len, enc_dim]
        :param time_enc: [batch_size, seq_len, enc_dim]
        :param causality: attention refers only to past items
        :return: [batch_size, feature_dim]
        """
        k_seq = seq
        if pos_enc is not None:
            k_seq = torch.cat([k_seq, pos_enc], dim=-1)
        if time_enc is not None:
            k_seq = torch.cat([k_seq, time_enc], dim=-1)
        key = self._k_proj(k_seq)
        scores = torch.sum(key * self._query, dim=-1)
        attentions = torch.unsqueeze(F.softmax(scores, dim=-1), dim=2)
        res = torch.sum(attentions * seq, dim=1)
        if return_att:
            return res, attentions
        return res


class MHGlobalQueryAttention(nn.Module):
    def __init__(self, d_model, input_dim,
                 n_head, dropout=0.1,
                 activation=nn.Identity,
                 share_query: bool = False,
                 **kwargs):
        super(MHGlobalQueryAttention, self).__init__()
        if share_query:
            self._global_query = torch.randn([d_model])
        else:
            self._global_query = None

        self.attention_heads = nn.ModuleList(
            [GlobalQueryAttention(d_model, input_dim, dropout, activation,
                                  global_query=self._global_query)
             for _ in range(n_head)])

    def forward(self, seq, pos_enc, time_enc, causality):
        return torch.stack([att(seq, pos_enc, time_enc, causality)
                            for att in self.attention_heads], dim=1)

