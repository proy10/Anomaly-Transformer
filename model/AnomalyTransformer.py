import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
from einops import rearrange


class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., activation="gelu"):
        super(MlpBlock, self).__init__()
        self.fn1 = nn.Linear(dim, hidden_dim)
        self.fn2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        x = self.fn1(x)
        x = self.dropout(self.activation(x))
        return self.dropout(self.fn2(x))


class MixerBlock(nn.Module):
    def __init__(self, num_patches, channels_mlp_dim, tokens_mlp_dim, hidden_dim, dropout=0.):
        super(MixerBlock, self).__init__()
        self.mlp_block_token = MlpBlock(num_patches, tokens_mlp_dim, dropout)
        self.mlp_block_chnl = MlpBlock(hidden_dim, channels_mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.norm1(x)  # (n_samples, n_patches, hidden_dim)
        y = rearrange(y, 'b n d -> b d n')  # (n_samples, hidden_dim, n_patches)
        y = self.mlp_block_token(y)  # (n_samples, hidden_dim, n_patches)
        y = rearrange(y, 'b d n -> b n d')  # (n_samples, n_patches, hidden_dim)
        x = x + y  # (n_samples, n_patches, hidden_dim)
        y = self.mlp_block_chnl(self.norm2(x))  # (n_samples, n_patches, hidden_dim)
        return x + y


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, series_length=None, patch_size=1, d_channel=2048, d_token=256, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.patch_embedding = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=patch_size,
            stride=patch_size
        )
        assert series_length % patch_size == 0
        num_patches = series_length // patch_size
        self.mixer_block = MixerBlock(
            num_patches=num_patches,
            channels_mlp_dim=d_channel,
            tokens_mlp_dim=d_token,
            hidden_dim=d_ff,
            dropout=dropout
        )
        self.conv1 = nn.Conv1d(in_channels=num_patches, out_channels=series_length, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)  # B, L, D
        # print(x.shape)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        # patch
        y = rearrange(y, "b n d -> b d n")  # B, D, L
        y = self.patch_embedding(y)  # (n_samples, hidden_dim, n_patches)
        y = rearrange(y, "b d n -> b n d")  # (n_samples, n_patches, hidden_dim)
        y = self.mixer_block(y)  # (n_samples, n_patches, hidden_dim)
        y = self.conv1(y)  # B, L, D
        y = rearrange(y, "b d n -> b n d")  # B, D, L
        y = self.conv2(y)  # B, D, L
        y = rearrange(y, "b d n -> b n d")  # B, L, D
        y = self.dropout(y)

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, use_RevIN=False):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    series_length=win_size,
                    patch_size=10,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.use_RevIN = use_RevIN
        if use_RevIN:
            self.revin_layer = RevIN(enc_in)

    def forward(self, x):
        if self.use_RevIN:
            x = self.revin_layer(x, 'norm')
        
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.use_RevIN:
            enc_out = self.revin_layer(enc_out, 'denorm')

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
