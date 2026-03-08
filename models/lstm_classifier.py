"""
Bidirectional LSTM with attention for power electronics fault classification.

Architecture:
  Input: (batch, window_size, n_channels)  [time-first format for LSTM]
  → 2-layer BiLSTM
  → Additive attention pooling over time steps
  → Dropout → Linear classifier

Bidirectional processing captures both past context (forward LSTM) and
future context (backward LSTM), which is appropriate for offline fault
analysis on fixed-length windows.  The attention mechanism learns to
focus on the most fault-discriminative time steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over LSTM hidden states.

    Produces a context vector as a weighted sum of hidden states,
    where weights are computed via a learnable scoring function.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden : torch.Tensor, shape (B, T, H)

        Returns
        -------
        context : torch.Tensor, shape (B, H)
        weights : torch.Tensor, shape (B, T)  — attention weights
        """
        scores = self.score(hidden).squeeze(-1)      # (B, T)
        weights = F.softmax(scores, dim=-1)           # (B, T)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)  # (B, H)
        return context, weights


class BiLSTMClassifier(nn.Module):
    """2-layer Bidirectional LSTM with attention for fault classification.

    Parameters
    ----------
    n_channels : int
        Number of input signal channels (input features per time step).
    n_classes : int
        Number of fault classes.
    hidden_size : int
        LSTM hidden state dimension (per direction).
    n_layers : int
        Number of LSTM layers (stacked).
    dropout : float
        Dropout between LSTM layers and before classifier.

    Input convention
    ----------------
    The model expects input shaped (B, C, T) — standard for this repo.
    Internally it transposes to (B, T, C) for the LSTM.

    Examples
    --------
    >>> model = BiLSTMClassifier(n_channels=6, n_classes=9)
    >>> x = torch.randn(32, 6, 1024)
    >>> logits = model(x)  # (32, 9)
    """

    def __init__(
        self,
        n_channels: int = 6,
        n_classes: int = 9,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # BiLSTM output is 2 × hidden_size
        self.attention = AdditiveAttention(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)

        Returns
        -------
        logits : torch.Tensor, shape (B, n_classes)
        """
        x = x.permute(0, 2, 1)  # (B, T, C) — LSTM time-first format
        hidden_states, _ = self.lstm(x)  # (B, T, 2*H)
        context, self._last_attention_weights = self.attention(hidden_states)
        return self.classifier(context)

    def forward_with_attention(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns attention weights for visualization.

        Returns
        -------
        logits : torch.Tensor, shape (B, n_classes)
        weights : torch.Tensor, shape (B, T)
        """
        x = x.permute(0, 2, 1)
        hidden_states, _ = self.lstm(x)
        context, weights = self.attention(hidden_states)
        logits = self.classifier(context)
        return logits, weights

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention context vector before the classifier."""
        x = x.permute(0, 2, 1)
        hidden_states, _ = self.lstm(x)
        context, _ = self.attention(hidden_states)
        return context

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps with long sequences)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
