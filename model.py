import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        r = x
        y = F.relu(self.conv(x.transpose(1,2))).transpose(1,2)
        y = self.drop(y)
        return self.norm(y + r)

class Encoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        d_model: int = 80,
        gru_hidden: int = 80,
        attn_heads: int = 4,
        emb_dim: int = 160,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.spatial = nn.Conv1d(in_ch, d_model, kernel_size=1)
        self.tblock = TemporalConvBlock(d_model, dropout)
        self.gru = nn.GRU(d_model, gru_hidden, batch_first=True, bidirectional=True)

        rnn_out = gru_hidden * 2
        self.attn = nn.MultiheadAttention(rnn_out, attn_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(rnn_out)

        self.head = nn.Sequential(
            nn.Linear(rnn_out, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x = self.spatial(x.transpose(1,2)).transpose(1,2)  # (B,L,d)
        x = self.tblock(x)
        x, _ = self.gru(x)  # (B,L,2h)
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + a)

        z = self.head(x.mean(dim=1))  # (B, emb_dim)
        return F.normalize(z, dim=-1)

class Projector(nn.Module):
    def __init__(self, emb_dim: int = 160, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)
