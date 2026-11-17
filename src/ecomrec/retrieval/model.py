"""Two-tower model: a shared item encoder, an item tower and a user tower."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ecomrec.retrieval.features import N_AGE_BUCKETS, N_CHANNEL_BUCKETS, N_GAP_BUCKETS


class ItemEncoder(nn.Module):
    """Concatenated embeddings of an item's id and categorical attributes."""

    def __init__(self, item_features: Tensor, cardinalities: list[int], dims: list[int]) -> None:
        super().__init__()
        self.register_buffer("item_features", item_features)
        self.embeddings = nn.ModuleList(
            nn.Embedding(n, d, padding_idx=0) for n, d in zip(cardinalities, dims, strict=True)
        )
        self.out_dim = sum(dims)

    def forward(self, rows: Tensor, id_dropout: float = 0.0) -> Tensor:
        codes = self.item_features[rows]
        ids = codes[..., 0]
        if self.training and id_dropout > 0:
            # Hide the id sometimes so the attribute-only path (used for unseen items) trains.
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < id_dropout, 0)
        parts = [self.embeddings[0](ids)]
        parts += [emb(codes[..., j]) for j, emb in enumerate(self.embeddings) if j > 0]
        return torch.cat(parts, dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        item_features: Tensor,
        cardinalities: list[int],
        dim: int = 64,
        hidden: int = 256,
        id_dim: int = 64,
        product_dim: int = 32,
        attr_dim: int = 8,
    ) -> None:
        super().__init__()
        dims = [id_dim, product_dim] + [attr_dim] * (len(cardinalities) - 2)
        self.encoder = ItemEncoder(item_features, cardinalities, dims)
        d = self.encoder.out_dim
        self.age = nn.Embedding(N_AGE_BUCKETS, 8)
        self.channel = nn.Embedding(N_CHANNEL_BUCKETS, 4)
        self.gap = nn.Embedding(N_GAP_BUCKETS, 8)
        self.item_mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        # user input: mean of history, most recent item, age, channel preference, recency
        self.user_mlp = nn.Sequential(
            nn.Linear(2 * d + 8 + 4 + 8, hidden), nn.ReLU(), nn.Linear(hidden, dim)
        )

    def item_vectors(self, rows: Tensor, id_dropout: float = 0.0) -> Tensor:
        return F.normalize(self.item_mlp(self.encoder(rows, id_dropout)), dim=-1)

    def user_vectors(self, history: Tensor, age: Tensor, channel: Tensor, gap: Tensor) -> Tensor:
        """``history`` is ``[batch, length]`` item rows, most recent first, 0-padded."""
        encoded = self.encoder(history)
        mask = (history > 0).unsqueeze(-1)
        pooled = (encoded * mask).sum(1) / mask.sum(1).clamp(min=1)
        x = torch.cat(
            [pooled, encoded[:, 0], self.age(age), self.channel(channel), self.gap(gap)], dim=-1
        )
        return F.normalize(self.user_mlp(x), dim=-1)


def in_batch_softmax_loss(
    users: Tensor, items: Tensor, item_rows: Tensor, log_q: Tensor | None, temperature: float
) -> Tensor:
    """Sampled softmax with in-batch negatives.

    ``log_q`` is the log sampling probability of each batch item. Subtracting it (the logQ
    correction) undoes the bias of in-batch sampling, where popular items show up as
    negatives in proportion to their popularity. Other rows holding the same item as a
    row's positive are masked rather than treated as negatives.
    """
    logits = users @ items.T / temperature
    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)
    same = item_rows.unsqueeze(0) == item_rows.unsqueeze(1)
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(users), device=users.device))
