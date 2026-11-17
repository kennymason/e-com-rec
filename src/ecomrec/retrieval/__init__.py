"""Stage 1: two-tower retrieval.

torch and faiss must not be imported in the same process on macOS (each bundles its own
OpenMP runtime and the second one aborts). So the stage is split:

- ``features``, ``model``, ``train``: torch. Training exports embeddings as ``.npy`` files.
- ``index``: faiss and numpy only. Builds the index from the exported embeddings.

Serving follows the same split: the user tower is exported for a torch-free runtime.
"""
