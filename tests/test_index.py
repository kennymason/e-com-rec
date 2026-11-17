import numpy as np
import pytest

pytestmark = pytest.mark.faiss


def test_flat_index_matches_brute_force():
    # Imported here so that collecting this file never loads faiss next to torch.
    from ecomrec.retrieval.index import ItemIndex

    rng = np.random.default_rng(0)
    items = rng.normal(size=(500, 16)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    article_ids = np.arange(1000, 1500)
    queries = items[:20] + 0.01 * rng.normal(size=(20, 16)).astype(np.float32)

    ids, scores = ItemIndex(items, article_ids).search(queries, k=10)
    expected = article_ids[np.argsort(-(queries @ items.T), axis=1)[:, :10]]
    assert ids.tolist() == expected.tolist()
    assert ids[:, 0].tolist() == article_ids[:20].tolist()
    assert (np.diff(scores, axis=1) <= 0).all()


def test_k_is_capped_at_the_index_size():
    from ecomrec.retrieval.index import ItemIndex

    items = np.eye(3, dtype=np.float32)
    ids, _ = ItemIndex(items, np.array([7, 8, 9])).search(items[:1], k=10)
    assert ids.shape == (1, 3)
    assert ids[0, 0] == 7 and sorted(ids[0].tolist()) == [7, 8, 9]
