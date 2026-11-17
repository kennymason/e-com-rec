"""Training and inference for the two-tower retriever (torch)."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ecomrec.retrieval.features import (
    GAP_EDGES,
    ItemTable,
    age_buckets,
    build_interactions,
    build_item_table,
)
from ecomrec.retrieval.model import TwoTower, in_batch_softmax_loss

EPOCH = date(1970, 1, 1)


@dataclass
class TwoTowerConfig:
    dim: int = 64
    hidden: int = 256
    history_len: int = 20
    train_days: int = 28
    """Purchases in the last ``train_days`` before the cutoff are the training positives.
    Each one's user history is read from all earlier days, however old."""
    index_days: int | None = 14
    """Only index articles sold in the last ``index_days``. None indexes every article with
    a sale in history."""
    epochs: int = 6
    batch_size: int = 4096
    lr: float = 1e-3
    temperature: float = 0.05
    logq: bool = True
    id_dropout: float = 0.1
    seed: int = 0


def pick_device() -> torch.device:
    """CUDA if present, otherwise CPU. Apple's MPS backend is deliberately not picked: this
    model is mostly embedding lookups and an epoch measured about 3x slower on MPS than on
    CPU (M-series laptop, 10% sample)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoTowerRetriever:
    """Same interface as the baselines: ``fit(history, cutoff)`` then ``recommend``."""

    def __init__(
        self,
        articles: pl.DataFrame,
        customers: pl.DataFrame,
        config: TwoTowerConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.articles = articles
        self.config = config or TwoTowerConfig()
        self.device = device or pick_device()
        self.age = torch.from_numpy(age_buckets(customers)).to(self.device)

    def fit(
        self,
        history: pl.DataFrame,
        cutoff: date,
        on_epoch: Callable[[int, float], None] | None = None,
    ) -> "TwoTowerRetriever":
        cfg, dev = self.config, self.device
        torch.manual_seed(cfg.seed)
        history = history.filter(pl.col("t_dat") < cutoff)
        cutoff_day = (cutoff - EPOCH).days

        self.items: ItemTable = build_item_table(
            self.articles, history["article_id"].unique().to_numpy()
        )
        inter = build_interactions(history, self.items, len(self.age))
        self.item_row = torch.from_numpy(inter.item_row.astype(np.int32)).to(dev)
        self.online = torch.from_numpy(inter.online.astype(np.uint8)).to(dev)
        self.day = torch.from_numpy(inter.day.astype(np.int32)).to(dev)
        self.cutoff_day = cutoff_day
        self.first_row = torch.from_numpy(inter.first_row).to(dev)
        self.end_row = torch.from_numpy(inter.end_row).to(dev)

        train_rows = np.flatnonzero(inter.day >= cutoff_day - cfg.train_days)
        if len(train_rows) < cfg.batch_size:
            raise ValueError(f"only {len(train_rows)} training rows for batch {cfg.batch_size}")
        counts = np.bincount(inter.item_row[train_rows], minlength=len(self.items.features))
        log_q = torch.from_numpy(np.log(np.maximum(counts, 1) / len(train_rows))).float().to(dev)

        # Per training example: the item, the customer, and where their earlier history ends.
        pos_item = torch.from_numpy(inter.item_row[train_rows]).to(dev)
        pos_customer = torch.from_numpy(inter.customer[train_rows]).to(dev)
        pos_end = torch.from_numpy(inter.day_start[train_rows]).to(dev)
        pos_start = torch.from_numpy(inter.cust_start[train_rows]).to(dev)
        pos_day = torch.from_numpy(inter.day[train_rows].astype(np.int64)).to(dev)

        self.model = TwoTower(
            torch.from_numpy(self.items.features),
            self.items.cardinalities,
            dim=cfg.dim,
            hidden=cfg.hidden,
        ).to(dev)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        recent_days = cfg.index_days if cfg.index_days is not None else cutoff_day
        recent = inter.item_row[inter.day >= cutoff_day - recent_days]
        self.index_rows = torch.from_numpy(np.unique(recent)).to(dev)

        n_batches = len(train_rows) // cfg.batch_size
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            perm = torch.randperm(len(train_rows), device=dev)
            total = 0.0
            for b in range(n_batches):
                batch = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]
                hist, channel, gap = self._history(pos_end[batch], pos_start[batch], pos_day[batch])
                users = self.model.user_vectors(hist, self.age[pos_customer[batch]], channel, gap)
                rows = pos_item[batch]
                items = self.model.item_vectors(rows, cfg.id_dropout)
                loss = in_batch_softmax_loss(
                    users, items, rows, log_q[rows] if cfg.logq else None, cfg.temperature
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            self._refresh_item_vectors()
            if on_epoch is not None:
                on_epoch(epoch, total / n_batches)
        return self

    def _history(self, ends: torch.Tensor, starts: torch.Tensor, target_day: torch.Tensor | int):
        """History windows ending just before row ``ends``, not reaching before ``starts``.
        Returns item rows ``[batch, history_len]`` (most recent first), channel buckets, and
        buckets of the days between the most recent purchase and ``target_day``."""
        steps = torch.arange(1, self.config.history_len + 1, device=self.device)
        offsets = ends.unsqueeze(1) - steps
        valid = offsets >= starts.unsqueeze(1)
        offsets = offsets.clamp(min=0)
        hist = torch.where(valid, self.item_row[offsets].long(), 0)
        n = valid.sum(1)
        share = (self.online[offsets] * valid).sum(1) / n.clamp(min=1)
        channel = torch.where(share < 1 / 3, 1, torch.where(share <= 2 / 3, 2, 3))
        days = target_day - self.day[offsets[:, 0]].long()
        gap = torch.bucketize(days, torch.tensor(GAP_EDGES, device=self.device), right=True) + 1
        return hist, torch.where(n == 0, 0, channel), torch.where(n == 0, 0, gap)

    @torch.no_grad()
    def _refresh_item_vectors(self) -> None:
        self.model.eval()
        chunks = [self.model.item_vectors(rows) for rows in self.index_rows.split(8192)]
        self.item_vectors = torch.cat(chunks)

    @torch.no_grad()
    def user_vectors(self, customers: np.ndarray) -> torch.Tensor:
        """User embeddings from each customer's history up to the cutoff."""
        self.model.eval()
        out = []
        for chunk in torch.from_numpy(customers.astype(np.int64)).to(self.device).split(8192):
            hist, channel, gap = self._history(
                self.end_row[chunk], self.first_row[chunk], self.cutoff_day
            )
            out.append(self.model.user_vectors(hist, self.age[chunk], channel, gap))
        return torch.cat(out)

    @torch.no_grad()
    def recommend(self, customers: pl.Series, n: int) -> pl.DataFrame:
        """Exact inner-product search in torch. ``retrieval.index`` does the same with FAISS
        from the exported embeddings."""
        users = self.user_vectors(customers.to_numpy())
        n = min(n, len(self.index_rows))
        top = [(chunk @ self.item_vectors.T).topk(n).indices for chunk in users.split(4096)]
        rows = self.index_rows[torch.cat(top)].cpu().numpy()
        return pl.DataFrame(
            {"customer_idx": customers, "article_id": self.items.article_ids[rows - 1]},
            schema_overrides={"article_id": pl.List(pl.Int32)},
        )

    def export(self, out_dir: Path, customers: pl.Series) -> None:
        """Write the embeddings the FAISS stage needs, and the model weights."""
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = self.index_rows.cpu().numpy()
        np.save(out_dir / "item_vectors.npy", self.item_vectors.cpu().numpy())
        np.save(out_dir / "item_article_ids.npy", self.items.article_ids[rows - 1])
        np.save(out_dir / "user_vectors.npy", self.user_vectors(customers.to_numpy()).cpu().numpy())
        np.save(out_dir / "user_customer_idx.npy", customers.to_numpy())
        torch.save(
            {"state_dict": self.model.state_dict(), "config": asdict(self.config)},
            out_dir / "model.pt",
        )
