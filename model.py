import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from RevIN.RevIN import RevIN

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils import TriangularCausalMask, ProbMask
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import os


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads=1, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.vis = False
        self.num_attention_heads = 1
        self.hidden_size = 1824
        self.hidden_size2 = 1680
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attention_head_size2 = int(self.hidden_size2 / self.num_attention_heads)
        self.all_head_size2 = self.num_attention_heads * self.attention_head_size2

        self.query = Linear(self.hidden_size, self.all_head_size)
        self.key = Linear(self.hidden_size, self.all_head_size)
        self.value = Linear(self.hidden_size, self.all_head_size)
        self.query2 = Linear(self.hidden_size2, self.all_head_size)
        self.key2 = Linear(self.hidden_size2, self.all_head_size)
        self.value2 = Linear(self.hidden_size2, self.all_head_size)

        self.out = Linear(self.hidden_size, self.hidden_size)
        self.out2 = Linear(self.hidden_size2, self.hidden_size2)
        self.attn_dropout = Dropout(0)
        self.proj_dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(-1,1,1824)
        return x.permute(0, 1, 2)
    def transpose_for_scores2(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size2)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 2)

    def forward(self, x1,x2):
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x1)
        mixed_value_layer = self.value(x1)


        mixed_query_layer2 = self.query2(x2)
        mixed_key_layer2 = self.key2(x2)
        mixed_value_layer2 = self.value2(x2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        a = key_layer.transpose(-1, -2)
        a2 = key_layer2.transpose(-1, -2)

        if query_layer.shape[0] != key_layer2.shape[0]:
            return
        # i = 0
        # i +=1
        # print(i)
        attention_scores = torch.matmul(query_layer, key_layer2.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_scores2 = torch.matmul(query_layer2, key_layer.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_probs2 = self.softmax(attention_scores2)


        eps = 1e-10
        batch_size = key_layer.size(0)
        patch = key_layer




        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer2)
        context_layer = context_layer.permute(0, 1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        weights = attention_probs2 if self.vis else None
        attention_probs2 = self.attn_dropout(attention_probs2)
        context_layer2 = torch.matmul(attention_probs2, value_layer)
        context_layer2 = context_layer2.permute(0, 1, 2).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)
        attention_output2 = self.out(context_layer2)
        attention_output2 = self.proj_dropout(attention_output2)

        out = torch.concat([attention_output,attention_output2],dim=-1)
        return out


    # def forward(self, hidden_states):
    #     hidden_states = hidden_states.view(hidden_states.size(0), -1)
    #     mixed_query_layer = self.query(hidden_states)
    #     mixed_key_layer = self.key(hidden_states)
    #     mixed_value_layer = self.value(hidden_states)
    #
    #     query_layer = self.transpose_for_scores(mixed_query_layer)
    #     key_layer = self.transpose_for_scores(mixed_key_layer)
    #     value_layer = self.transpose_for_scores(mixed_value_layer)
    #     a = key_layer.transpose(-1, -2)
    #     attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    #     attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    #     attention_probs = self.softmax(attention_scores)
    #
    #
    #     eps = 1e-10
    #     batch_size = key_layer.size(0)
    #     patch = key_layer
    #
    #
    #
    #
    #     weights = attention_probs if self.vis else None
    #     attention_probs = self.attn_dropout(attention_probs)
    #     context_layer = torch.matmul(attention_probs, value_layer)
    #     context_layer = context_layer.permute(0, 1, 2).contiguous()
    #     new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    #     context_layer = context_layer.view(*new_context_layer_shape)
    #     attention_output = self.out(context_layer)
    #     attention_output = self.proj_dropout(attention_output)
    #
    #     return attention_output

def centropy(x,y):
    n = 1e-6
    res = -np.sum(np.nan_to_num(x*np.log(y+n)),axis=1)
    return res

class mymodel(nn.Module):
    def __init__(self, d_model=24, dropout=0.1, nhead=8, nlayers=2, max_len=500) -> None:
        super().__init__()
        self.max_len = max_len
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, 512, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, 1)
        # self.decoder = nn.Linear(6, 1)
        self.init_weights()

        self.downConv = nn.Conv1d(in_channels=70,
                               out_channels=70,
                               kernel_size=3,
                               padding=2,
                               padding_mode='circular')
        self.Conv1 = nn.Conv1d(in_channels=70,
                              out_channels=1024,
                              kernel_size=3,
                              padding=1,
                              padding_mode='circular')
        self.Conv2 = nn.Conv1d(in_channels=1024,
                              out_channels=70,
                              kernel_size=3,
                              padding=1,
                              padding_mode='circular')
        self.norm1 = nn.BatchNorm1d(1024)
        self.norm2 = nn.BatchNorm1d(70)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, key_msk, attn_msk=None) -> Tensor:
        """
        return:
            output1: Tensor, extracted features
            output2: Tensor, predicted series
        """
        # reverse_layer
        # revein_layer = RevIN(src.shape[2])
        # re_s_in = revein_layer(src.cuda(), 'norm')

        src = self.pos_encoder(src.cuda())




        output1 = self.transformer_encoder(src, attn_msk, key_msk)
        output1 = self.dropout(output1)

        fea = self.Conv1(output1)
        fea = self.norm1(fea)
        fea = self.activation(fea)
        fea = self.maxPool(fea)

        fea = self.Conv2(fea)
        fea = self.norm2(fea)
        fea = self.activation(fea)
        x = self.maxPool(fea)

        # output2 = self.decoder(x)
        # 不使用convrp
        # rul = self.decoder(x)
        # output1 = revein_layer(output1, 'denorm')




        rul = self.decoder(output1)


        # return output1, rul,rul

        return output1, rul,x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, feature_num]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)
        

class Discriminator(nn.Module): #D_y
    def __init__(self, in_features=24) -> None:
        super().__init__()
        self.in_features = in_features
        self.li = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.downConv = nn.Conv1d(in_channels=1,
                                  out_channels=1,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        # self.Conv1 = nn.Conv1d(in_channels=max_len,
        #                       out_channels=1024,
        #                       kernel_size=3,
        #                       padding=1,
        #                       padding_mode='circular')
        # self.Conv1 = nn.Conv1d(in_channels=1024,
        #                       out_channels=max_len,
        #                       kernel_size=3,
        #                       padding=1,
        #                       padding_mode='circular')
        self.norm = nn.BatchNorm1d(1)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        x: Tensor, shape [bts, in_features]
        """
        # x = x.unsqueeze(dim=2)
        # fea = self.downConv(x.permute(0, 2, 1))
        # fea = self.norm(fea)
        # fea = self.activation(fea)
        # fea = self.maxPool(fea).transpose(1, 2)
        # fea = fea.squeeze()

        x = ReverseLayer.apply(x, 1)
        if x.size(0) == 1:
            pad = torch.zeros(1, self.in_features).cuda()
            x = torch.cat((x, pad), 0)
            y = self.li(x)[0].unsqueeze(0)
            return y
        return self.li(x)


class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
        

class backboneDiscriminator(nn.Module): #D_f
    def __init__(self, seq_len, d_model=24) -> None:
        super().__init__()
        self.seq_len = seq_len
        # self.li1 = nn.Linear(d_model, 1)
        # self.li1 = nn.Linear(6, 1)
        # self.li1 = nn.Linear(70, 1)

        # only t
        # self.li1 = nn.Linear(24, 1)
        # self.li2 = nn.Sequential(
        #     nn.Linear(70, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        #     nn.Sigmoid()
        # )

        # p+t
        # self.li1 = nn.Linear(292, 1)
        self.li1 = nn.Linear(146, 1)
        self.li2 = nn.Sequential(
            nn.Linear(24, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.downConv = nn.Conv1d(in_channels=d_model,
                               out_channels=d_model,
                               kernel_size=3,
                               padding=2,
                               padding_mode='circular')
        self.Conv1 = nn.Conv1d(in_channels=d_model,
                              out_channels=1024,
                              kernel_size=3,
                              padding=1,
                              padding_mode='circular')
        self.Conv2 = nn.Conv1d(in_channels=1024,
                              out_channels=d_model,
                              kernel_size=3,
                              padding=1,
                              padding_mode='circular')
        self.norm1 = nn.BatchNorm1d(1024)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.att = Attention()
    # def forward(self, x):
    #
    #     x = ReverseLayer.apply(x, 1)
    #     # attention
    #     src2 = self.att(x).reshape(-1,24,146)
    #     src3 = torch.concat([x,src2],dim=-1)
    #     out1 = self.li1(src3).squeeze(2)
    #     if x.size(0) == 1:
    #         pad = torch.zeros(1, 70).cuda()
    #         out1 = torch.cat((out1, pad), 0)
    #         out2 = self.li2(out1)[0].unsqueeze(0)
    #         return out2
    #     out2 = self.li2(out1)
    #     return out2

    def forward(self, x1,x2):

        x1 = ReverseLayer.apply(x1, 1)
        x2 = ReverseLayer.apply(x2, 1)
        # attention
        src2 = self.att(x1,x2)
        if src2 == None:
            return
        src2 = src2.reshape(-1,24,152)

        x = torch.concat([x1,x2],dim=-1)
        # src3 = torch.concat([x,src2],dim=-1)
        out1 = self.li1(x).squeeze(2)
        if x.size(0) == 1:
            pad = torch.zeros(1, 70).cuda()
            out1 = torch.cat((out1, pad), 0)
            out2 = self.li2(out1)[0].unsqueeze(0)
            return out2
        out2 = self.li2(out1)
        return out2


class patch_backboneDiscriminator(nn.Module): #D_f
    def __init__(self, seq_len, d=24) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.li1 = nn.Linear(seq_len, 1)
        self.li2 = nn.Sequential(
            nn.Linear(d, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.unsqueeze(dim=2)
        # fea = self.downConv(x.permute(0, 2, 1))
        # fea = self.norm(fea)
        # fea = self.activation(fea)
        # fea = self.maxPool(fea).transpose(1, 2)
        # fea = fea.squeeze()
        x = x.permute(0, 2, 1)
        x = ReverseLayer.apply(x, 1)
        out1 = self.li1(x).squeeze(2)
        if x.size(0) == 1:
            pad = torch.zeros(1, self.seq_len).cuda()
            out1 = torch.cat((out1, pad), 0)
            out2 = self.li2(out1)[0].unsqueeze(0)
            return out2
        out2 = self.li2(out1)
        return out2

if __name__ == "__main__":
    pass
    