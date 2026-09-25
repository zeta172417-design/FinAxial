from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def stock_vocab_sha256(codes: Sequence[object]) -> str:
    """Hash the ordered stock vocabulary stored alongside each checkpoint."""
    digest = hashlib.sha256()
    for code in codes:
        digest.update(str(code).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        inverse = 1.0 / (
            float(base) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = query.shape[-2]
        positions = torch.arange(length, device=query.device, dtype=torch.float32)
        frequency = torch.outer(positions, self.inverse_frequency.float())
        angles = torch.repeat_interleave(frequency, 2, dim=-1)
        cosine = angles.cos().to(dtype=query.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=query.dtype)[None, None, :, :]
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SlidingTemporalSelfAttention(nn.Module):
    """Causal temporal attention limited to a fixed trailing lookback."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        sequence_length: int,
        lookback: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = d_model // heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        positions = torch.arange(sequence_length)
        query = positions[:, None]
        key = positions[None, :]
        allowed = (key <= query) & (key >= query - int(lookback) + 1)
        bias = torch.zeros(sequence_length, sequence_length, dtype=torch.float32)
        bias.masked_fill_(~allowed, float("-inf"))
        self.register_buffer("attention_bias", bias, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stocks, length, width = values.shape
        if length != self.attention_bias.shape[0]:
            raise ValueError(f"expected sequence length {self.attention_bias.shape[0]}, got {length}")
        qkv = self.qkv(values).view(stocks, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rope(query, key)
        attended = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=self.attention_bias.to(dtype=query.dtype)[None, None, :, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.output(attended.transpose(1, 2).reshape(stocks, length, width))


class SlidingTemporalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        sequence_length: int,
        lookback: int,
        *,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = SlidingTemporalSelfAttention(
            d_model, heads, sequence_length, lookback, dropout=attention_dropout,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.residual_dropout(self.attention(self.attention_norm(values)))
        return values + self.ffn(self.ffn_norm(values))


class BatchedStockSelfAttention(nn.Module):
    """Shared stock attention applied to multiple dates as a batch."""

    def __init__(self, d_model: int, heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = d_model // heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        dates, stocks, width = values.shape
        eligible = eligible.reshape(dates, stocks).bool()
        attention_eligible = eligible.clone()
        empty = ~attention_eligible.any(dim=1)
        attention_eligible[empty, 0] = True
        qkv = self.qkv(values).view(dates, stocks, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        key_bias = torch.zeros(
            dates, stocks, device=values.device, dtype=values.dtype,
        ).masked_fill(~attention_eligible, float("-inf"))
        attended = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=key_bias[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(dates, stocks, width)
        return self.output(attended) * eligible[..., None].to(values.dtype)


class BatchedStockBlock(nn.Module):
    def __init__(
        self, d_model: int, heads: int, ffn_dim: int,
        *, dropout: float, attention_dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = BatchedStockSelfAttention(
            d_model, heads, dropout=attention_dropout,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        mask = eligible[..., None].to(values.dtype)
        values = values + self.residual_dropout(
            self.attention(self.attention_norm(values), eligible)
        )
        values = values * mask
        values = values + self.ffn(self.ffn_norm(values)) * mask
        return values * mask


class CausalMultiScaleMemory(nn.Module):
    """Attend to a few already-causal trend summaries at each date/stock."""

    def __init__(self, d_model: int, scales: int, channels: int) -> None:
        super().__init__()
        self.scales = int(scales)
        self.channels = int(channels)
        self.feature_projection = nn.Linear(channels, d_model)
        self.scale_embedding = nn.Parameter(torch.zeros(scales, d_model))
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.scale_embedding, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, recent: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if memory.shape != (*recent.shape[:2], self.scales, self.channels):
            raise ValueError("long memory shape does not match recent tokens")
        tokens = self.feature_projection(memory) + self.scale_embedding
        query = self.query(recent).unsqueeze(-2)
        keys = self.key(tokens)
        values = self.value(tokens)
        logits = (query * keys).sum(dim=-1) / (recent.shape[-1] ** 0.5)
        context = (torch.softmax(logits, dim=-1)[..., None] * values).sum(dim=-2)
        update = torch.sigmoid(self.gate(torch.cat((recent, context), dim=-1)))
        return recent + update * self.output(self.norm(context))


class StockTimeTransformer(nn.Module):
    """FinAxial: interleaved causal temporal and cross-sectional attention."""

    def __init__(
        self,
        stocks: int = 4650,
        lookback: int = 64,
        output_steps: int = 8,
        channels: int = 6,
        *,
        context_days: int | None = None,
        temporal_window: int | None = None,
        d_model: int = 96,
        heads: int = 4,
        temporal_layers: int = 2,
        stock_layers: int = 2,
        ffn_dim: int = 384,
        stock_embedding_dim: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        stock_id_dropout: float = 0.1,
        architecture: str = "stacked",
        head_mode: str = "single",
        return_scale: float = 0.02,
        long_memory_scales: Sequence[int] = (),
        long_memory_channels: int = 6,
    ) -> None:
        super().__init__()
        if stocks <= 0 or lookback <= 0 or output_steps <= 1 or channels <= 0:
            raise ValueError("stocks/lookback/channels must be positive and output_steps > 1")
        if not 0 <= stock_id_dropout < 1:
            raise ValueError("stock_id_dropout must be in [0, 1)")
        if architecture not in {"stacked", "interleaved_axial"}:
            raise ValueError("architecture must be 'stacked' or 'interleaved_axial'")
        if head_mode not in {"single", "dual"} or return_scale <= 0:
            raise ValueError("head mode must be single or dual, with positive return scale")
        if long_memory_scales and architecture != "interleaved_axial":
            raise ValueError("long memory currently needs interleaved axial architecture")
        if any(int(scale) < 2 for scale in long_memory_scales) or long_memory_channels <= 0:
            raise ValueError("long memory scales/channels are invalid")
        if architecture == "interleaved_axial" and (temporal_layers < 2 or stock_layers < 2):
            raise ValueError("interleaved_axial requires at least two temporal and stock layers")
        resolved_context = int(lookback) - 1 if context_days is None else int(context_days)
        resolved_window = int(lookback) if temporal_window is None else int(temporal_window)
        if resolved_context < 0 or resolved_window <= 0:
            raise ValueError("context_days must be non-negative and temporal_window positive")
        self.stocks = int(stocks)
        self.lookback = int(lookback)
        self.context_days = resolved_context
        self.temporal_window = resolved_window
        self.output_steps = int(output_steps)
        self.sequence_length = self.context_days + self.output_steps
        self.channels = int(channels)
        self.d_model = int(d_model)
        self.stock_id_dropout = float(stock_id_dropout)
        self.architecture = str(architecture)
        self.head_mode = str(head_mode)
        self.return_scale = float(return_scale)
        self.long_memory_scales = tuple(int(scale) for scale in long_memory_scales)
        self.long_memory_channels = int(long_memory_channels)
        self.unknown_stock_id = self.stocks

        self.feature_projection = nn.Linear(channels, d_model)
        self.stock_embedding = nn.Embedding(stocks + 1, stock_embedding_dim)
        self.stock_projection = nn.Linear(stock_embedding_dim, d_model, bias=False)
        self.identity_gate = nn.Linear(d_model, d_model)
        nn.init.normal_(self.stock_embedding.weight, mean=0.0, std=0.02)
        self.input_norm = nn.LayerNorm(d_model)
        self.temporal_blocks = nn.ModuleList([
            SlidingTemporalBlock(
                d_model, heads, ffn_dim, self.sequence_length, self.temporal_window,
                dropout=dropout, attention_dropout=attention_dropout,
            )
            for _ in range(temporal_layers)
        ])
        self.stock_blocks = nn.ModuleList([
            BatchedStockBlock(
                d_model, heads, ffn_dim,
                dropout=dropout, attention_dropout=attention_dropout,
            )
            for _ in range(stock_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.return_head = nn.Linear(d_model, 1, bias=False)
        if self.head_mode == "dual":
            self.absolute_return_norm = nn.LayerNorm(d_model)
            self.absolute_return_head = nn.Linear(d_model, 1, bias=False)
        if self.long_memory_scales:
            self.long_memory = CausalMultiScaleMemory(
                d_model, len(self.long_memory_scales), self.long_memory_channels,
            )
        self.register_buffer(
            "_default_stock_ids", torch.arange(stocks, dtype=torch.long), persistent=False,
        )

    def _stock_ids(self, stock_ids: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        ids = self._default_stock_ids if stock_ids is None else stock_ids
        ids = ids.to(device=device, dtype=torch.long).reshape(-1)
        if ids.numel() != self.stocks:
            raise ValueError(f"expected {self.stocks} stock ids, got {ids.numel()}")
        if bool(((ids < 0) | (ids > self.unknown_stock_id)).any()):
            raise ValueError("stock id is outside the configured vocabulary")
        if self.training and self.stock_id_dropout > 0:
            drop = torch.rand(ids.shape, device=device) < self.stock_id_dropout
            ids = torch.where(drop, self.unknown_stock_id, ids)
        return ids

    def encode_hidden(
        self,
        values: torch.Tensor,
        token_valid: torch.Tensor,
        eligible: torch.Tensor,
        stock_ids: torch.Tensor | None = None,
        long_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim == 4:
            if values.shape[0] != 1:
                raise ValueError("StockTimeTransformer consumes one full-market sequence per rank")
            values = values.squeeze(0)
            token_valid = token_valid.squeeze(0)
            eligible = eligible.squeeze(0)
            if long_memory is not None:
                long_memory = long_memory.squeeze(0)
        expected = (self.stocks, self.sequence_length, self.channels)
        if tuple(values.shape) != expected:
            raise ValueError(f"expected input {expected}, got {tuple(values.shape)}")
        token_valid = token_valid.to(values.device, dtype=torch.bool)
        eligible = eligible.to(values.device, dtype=torch.bool)
        if tuple(token_valid.shape) != (self.stocks, self.sequence_length):
            raise ValueError("token_valid has the wrong shape")
        if tuple(eligible.shape) != (self.output_steps, self.stocks):
            raise ValueError("eligible has the wrong shape")
        if self.long_memory_scales:
            expected_long = (
                self.stocks, self.sequence_length,
                len(self.long_memory_scales), self.long_memory_channels,
            )
            if long_memory is None or tuple(long_memory.shape) != expected_long:
                raise ValueError(f"expected long memory shape {expected_long}")
            long_memory = long_memory.to(device=values.device, dtype=values.dtype)
        elif long_memory is not None:
            raise ValueError("long memory was supplied to a model without a memory branch")

        hidden = self.feature_projection(values)
        ids = self._stock_ids(stock_ids, values.device)
        identity = self.stock_projection(self.stock_embedding(ids))[:, None, :]
        hidden = hidden + torch.sigmoid(self.identity_gate(hidden)) * identity
        hidden = self.input_norm(hidden)
        temporal_mask = token_valid[..., None].to(hidden.dtype)
        hidden = hidden * temporal_mask
        if self.architecture == "stacked":
            for block in self.temporal_blocks:
                hidden = block(hidden) * temporal_mask
            hidden = hidden[:, -self.output_steps:, :].permute(1, 0, 2).contiguous()
            for block in self.stock_blocks:
                hidden = block(hidden, eligible)
        else:
            full_stock_blocks = min(
                len(self.stock_blocks) - 1, len(self.temporal_blocks) - 1,
            )
            for index, block in enumerate(self.temporal_blocks):
                hidden = block(hidden) * temporal_mask
                if index == 0 and self.long_memory_scales:
                    hidden = self.long_memory(hidden, long_memory) * temporal_mask
                if index < full_stock_blocks:
                    by_date = hidden.permute(1, 0, 2).contiguous()
                    by_date = self.stock_blocks[index](by_date, token_valid.T)
                    hidden = by_date.permute(1, 0, 2).contiguous() * temporal_mask
            hidden = hidden[:, -self.output_steps:, :].permute(1, 0, 2).contiguous()
            for block in self.stock_blocks[full_stock_blocks:]:
                hidden = block(hidden, eligible)
        return hidden

    def score_hidden(
        self, hidden: torch.Tensor, eligible: torch.Tensor,
    ) -> torch.Tensor:
        """Project encoded supervised-date states to centered stock scores."""
        if tuple(hidden.shape) != (self.output_steps, self.stocks, self.d_model):
            raise ValueError(
                "expected hidden shape "
                f"{(self.output_steps, self.stocks, self.d_model)}, got {tuple(hidden.shape)}"
            )
        eligible = eligible.to(hidden.device, dtype=torch.bool)
        if tuple(eligible.shape) != (self.output_steps, self.stocks):
            raise ValueError("eligible has the wrong shape")
        prediction = self.return_head(self.output_norm(hidden)).squeeze(-1)
        mask = eligible.to(prediction.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (prediction * mask).sum(dim=1, keepdim=True) / count
        prediction = prediction - mean
        return torch.where(eligible, prediction, torch.zeros_like(prediction))

    def predict_return_hidden(
        self, hidden: torch.Tensor, eligible: torch.Tensor,
    ) -> torch.Tensor:
        """Absolute next-day return in decimal units, not a ranking proxy."""
        if self.head_mode != "dual":
            raise ValueError("absolute return head is available only in dual mode")
        if tuple(hidden.shape) != (self.output_steps, self.stocks, self.d_model):
            raise ValueError("hidden shape does not match the return head")
        prediction = self.absolute_return_head(
            self.absolute_return_norm(hidden)
        ).squeeze(-1) * self.return_scale
        return torch.where(
            eligible.to(hidden.device, dtype=torch.bool), prediction,
            torch.zeros_like(prediction),
        )

    def forward(
        self,
        values: torch.Tensor,
        token_valid: torch.Tensor,
        eligible: torch.Tensor,
        stock_ids: torch.Tensor | None = None,
        long_memory: torch.Tensor | None = None,
        return_heads: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(
            values, token_valid, eligible, stock_ids, long_memory,
        )
        if eligible.ndim == 3:
            eligible = eligible.squeeze(0)
        score = self.score_hidden(hidden, eligible)
        if return_heads:
            return score, self.predict_return_hidden(hidden, eligible)
        return score
