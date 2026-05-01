"""
Neural network model architectures for time-series forecasting.

Includes: CNN+LSTM, CNN+Transformer, PatchTST, Temporal Fusion
Transformer (simplified), and supporting components (positional
encoding, gated residual blocks, variable selection).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class SequenceBundle:
    """Holds scaled sequences and split masks for DL training."""
    X_all_scaled: np.ndarray
    y_all_scaled: np.ndarray
    X_sequences: np.ndarray
    y_sequences: np.ndarray
    sequence_target_positions: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


class WindowedDataset(Dataset):
    """PyTorch Dataset wrapping windowed (X, y) sequence arrays."""

    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_seq)

    def __getitem__(self, index: int):
        return self.X_seq[index], self.y_seq[index]


# ── Building blocks ──────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding added to token embeddings."""

    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        self.positional = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional[:, : x.size(1), :]


class GatedResidualBlock(nn.Module):
    """Gated Residual Network (GRN) block used in Temporal Fusion Transformers."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = F.elu(self.fc1(x))
        z = self.dropout(self.fc2(z))
        gated = torch.sigmoid(self.gate(x)) * z
        return self.norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """Soft feature selection via learned attention weights."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight_net = nn.Linear(n_features, n_features)
        self.proj = nn.Linear(n_features, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.weight_net(x), dim=-1)
        selected = x * weights
        return self.proj(selected), weights


# ── Model architectures ─────────────────────────────────────────────

class CNNLSTMRegressor(nn.Module):
    """1D-CNN feature extractor followed by a 2-layer LSTM."""

    def __init__(self, n_features: int, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :]).squeeze(-1)


class CNNTransformerRegressor(nn.Module):
    """1D-CNN encoder with Transformer self-attention layers."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.positional = LearnablePositionalEncoding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.positional(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class PatchTSTRegressor(nn.Module):
    """Patch Time-Series Transformer — patches the input before attention."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 6,
        stride: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.num_patches = 1 + max(0, (seq_len - patch_len) // stride)
        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.positional = LearnablePositionalEncoding(max(self.num_patches, 1), d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = []
        for start in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patch = x[:, start : start + self.patch_len, :].reshape(x.size(0), -1)
            patches.append(patch)
        patch_tensor = torch.stack(patches, dim=1)
        patch_tensor = self.patch_proj(patch_tensor)
        patch_tensor = self.positional(patch_tensor)
        encoded = self.encoder(patch_tensor)
        return self.head(encoded.mean(dim=1)).squeeze(-1)


class TemporalFusionTransformerRegressor(nn.Module):
    """Simplified TFT with variable selection, GRN, LSTM, and multi-head attention."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variable_selection = VariableSelectionNetwork(n_features, d_model)
        self.pre_grn = GatedResidualBlock(d_model, d_model * 2, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.post_grn = GatedResidualBlock(d_model, d_model * 2, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected, _ = self.variable_selection(x)
        encoded = self.pre_grn(selected)
        lstm_out, _ = self.lstm(encoded)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, need_weights=False)
        fused = self.post_grn(lstm_out + attn_out)
        return self.head(fused[:, -1, :]).squeeze(-1)


# ── Model factory ────────────────────────────────────────────────────

def build_sequence_model(
    model_name: str,
    n_features: int,
    seq_len: int,
    config: Dict,
) -> nn.Module:
    """Instantiate a sequence model by name using hyperparameters from *config*."""
    if model_name == "CNN + LSTM":
        return CNNLSTMRegressor(n_features=n_features, d_model=config["transformer_d_model"])
    if model_name == "PatchTST":
        return PatchTSTRegressor(
            n_features=n_features,
            seq_len=seq_len,
            patch_len=config["patch_len"],
            stride=config["patch_stride"],
            d_model=config["transformer_d_model"],
            n_heads=config["transformer_heads"],
            n_layers=config["transformer_layers"],
        )
    if model_name == "Temporal Fusion Transformer":
        return TemporalFusionTransformerRegressor(
            n_features=n_features,
            d_model=config["transformer_d_model"],
            n_heads=config["transformer_heads"],
        )
    if model_name == "CNN + Transformer":
        return CNNTransformerRegressor(
            n_features=n_features,
            seq_len=seq_len,
            d_model=config["transformer_d_model"],
            n_heads=config["transformer_heads"],
            n_layers=config["transformer_layers"],
        )
    raise ValueError(f"Unsupported sequence model: {model_name}")
