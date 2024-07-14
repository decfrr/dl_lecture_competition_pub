import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class AdvancedConvClassifier(nn.Module):
    def __init__(self, num_classes: int, num_subjects: int, in_channels: int, seq_len: int, hid_dim: int = 256,
                 num_layers: int = 4, p_drop: float = 0.5, l2_reg: float = 1e-4) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            *[ConvBlock(in_channels if i == 0 else hid_dim, hid_dim, p_drop=p_drop) for i in range(num_layers)]
        )

        self.attention = SelfAttention(hid_dim)

        self.subject_embedding = nn.Embedding(num_subjects, hid_dim)
        self.avg_subject_embedding = nn.Parameter(torch.zeros(hid_dim))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 2, num_classes),
        )

        self.l2_reg = l2_reg

    def forward(self, X: torch.Tensor, subject_ids: torch.Tensor = None) -> torch.Tensor:
        X = self.blocks(X)
        X = self.attention(X.transpose(1, 2)).transpose(1, 2)

        if subject_ids is not None:
            subject_emb = self.subject_embedding(subject_ids).unsqueeze(2)
            subject_emb = subject_emb.expand(-1, -1, X.size(2))
        else:
            subject_emb = self.avg_subject_embedding.unsqueeze(0).unsqueeze(2).expand(X.size(0), -1, X.size(2))

        X = torch.cat((X, subject_emb), dim=1)
        return self.head(X)

    def l2_regularization(self):
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return self.l2_reg * l2_reg


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.5) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_res = X
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))

        if self.in_dim == self.out_dim:
            X += X_res  # skip connection

        return self.dropout(X)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        b, t, d = X.shape
        qkv = self.to_qkv(X).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, t.size(1), self.heads, d // self.heads).transpose(1, 2), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.to_out(out)
