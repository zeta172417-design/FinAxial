from __future__ import annotations

import torch
from torch import nn

from third_party.StockMixer.src.model import StockMixer


class StockMixerReturn(nn.Module):
    """StockMixer adapted to output one raw return per stock directly."""

    def __init__(
        self,
        stocks: int = 4650,
        lookback: int = 32,
        channels: int = 6,
        market_hidden: int = 20,
        scale: int = 3,
        disable_stock_mixing: bool = False,
    ):
        super().__init__()
        self.stocks = stocks
        self.disable_stock_mixing = bool(disable_stock_mixing)
        self.backbone = StockMixer(stocks, lookback, channels, market_hidden, scale)
        if self.disable_stock_mixing:
            self.backbone.stock_mixer.requires_grad_(False)
            self.backbone.time_fc_.requires_grad_(False)

    def forward(self, values: torch.Tensor, eligible: torch.Tensor | None = None) -> torch.Tensor:
        if values.ndim == 4:
            if values.size(0) != 1:
                raise ValueError("StockMixer consumes one full-market date at a time")
            values = values.squeeze(0)
        if values.shape[0] != self.stocks:
            raise ValueError(f"expected {self.stocks} stocks, got {values.shape[0]}")
        if eligible is not None:
            eligible = eligible.reshape(-1).to(values.dtype)
            values = values * eligible[:, None, None]
        if not self.disable_stock_mixing:
            return self.backbone(values).squeeze(-1)
        # Preserve the upstream indicator/temporal branch and remove only the
        # cross-stock branch.  This is the planned architectural control.
        convolved = self.backbone.conv(values.permute(0, 2, 1)).permute(0, 2, 1)
        temporal = self.backbone.mixer(values, convolved)
        temporal = self.backbone.channel_fc(temporal).squeeze(-1)
        return self.backbone.time_fc(temporal).squeeze(-1)
