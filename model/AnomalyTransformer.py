import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from math import sqrt
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_ref_data(data_path, win_size, step):
    data = np.load(data_path + "/MSL_train.npy")
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data


class EncoderLayer(nn.Module):
    def __init__(self, attention, ref_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.ref_attention = ref_attention
        self.multiheadattn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, ref_x, attn_mask=None):
        new_x, attn, prior, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        new_x = new_x + self.dropout(new_x)
        new_x = self.norm0(new_x)

        new_ref_x, ref_attn, ref_prior, ref_sigma = self.ref_attention(
            ref_x, ref_x, ref_x,
            attn_mask=attn_mask
        )
        # B, L, H, E = ref_x.shape
        # _, S, _, D = x.shape
        # scale = 1. / sqrt(E)
        # scores = torch.einsum("blhe,bshe->bhls", new_ref_x, new_x)
        # series = self.dropout(torch.softmax(scale * scores, dim=-1))
        # new_x = torch.einsum("bhls,bshd->blhd", series, new_x)

        new_x, _ = self.multiheadattn(new_ref_x, new_x, new_x)

        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        ref_x = ref_x + self.dropout(new_ref_x)
        ref_y = ref_x = self.norm3(ref_x)
        ref_y = self.dropout(self.activation(self.conv3(ref_y.transpose(-1, 1))))
        ref_y = self.dropout(self.conv4(ref_y).transpose(-1, 1))

        return self.norm2(x + y), self.norm4(ref_x + ref_y), attn, prior, sigma, ref_attn, ref_prior


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, ref_x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        ref_series_list = []
        ref_prior_list = []
        for attn_layer in self.attn_layers:
            x, ref_x, series, prior, sigma, ref_series, ref_prior = attn_layer(x, ref_x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            ref_series_list.append(ref_series)
            ref_prior_list.append(ref_prior)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list, ref_series_list, ref_prior_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.ref_emmbedding = DataEmbedding(enc_in, d_model, dropout)
        self.ref_x = load_ref_data()

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        ref_x_emb = self.ref_emmbedding(self.ref_x)
        enc_out, series, prior, sigmas, ref_series, ref_prior = self.encoder(enc_out, ref_x_emb)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas, ref_series, ref_prior
        else:
            return enc_out  # [B, L, D]
