import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import TriangularCausalMask, ProbMask, fix_randomness
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np

fix_randomness(2)
class informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self,pred_len,output_attention,embed_type,enc_in,embed,dropout,dec_in,freq,e_layers,factor,d_model,n_heads,distil,d_layers,c_out):
        super(informer, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.embed_type = embed_type
        self.enc_in = enc_in
        self.embed = embed
        self.dropout = dropout
        self.dec_in = dec_in
        self.freq = freq
        self.e_layers = e_layers
        self.factor = factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.distil = distil
        self.d_layers = d_layers
        self.c_out = c_out
        # Embedding
        if self.embed_type == 0:
            self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                            self.dropout)
            self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        elif self.embed_type == 1:
            self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        elif self.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                               self.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq,
                                               self.dropout)

        elif self.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(self.enc_in, self.d_model, self.embed, self.freq,
                                               self.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(self.dec_in, self.d_model, self.embed, self.freq,
                                               self.dropout)
        elif self.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(self.enc_in, self.d_model, self.embed, self.freq,
                                               self.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(self.dec_in, self.d_model, self.embed, self.freq,
                                               self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_model*2,
                    dropout=self.dropout,
                    activation='gelu'
                ) for l in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_model*2,
                    dropout=self.dropout,
                    activation='gelu',
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )
        self.hidden_dim = 24
        self.dropout = 0.5
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim // 2, 1))

    def forward(self, x_enc,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # 32*96*512
        enc_out = self.enc_embedding(x_enc)
        # 32*49*512
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        x_dec = x_enc
        dec_out = self.dec_embedding(x_dec)
        dec_out,raw_feature = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        pre = self.regressor(dec_out)
        if self.output_attention:
            return dec_out, attns
        else:
            return enc_out,pre  # [B, L, D]
