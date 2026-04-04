"""
VELM Training — Multi-Dataset Curriculum Data Loader

Streams and tokenizes text from multiple HuggingFace datasets,
producing chunked token arrays for CALM autoencoder training.

Supports curriculum weighting to balance domain distribution:
  - math (OpenWebMath)
  - general text (WikiText-103)
  - narrative (TinyStories)

Usage:
  loader = CurriculumDataLoader(tokenizer, chunk_size=4)
  loader.load(target_chunks=250_000)
  batch = loader.get_batch(key, batch_size=64)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Default dataset sources and their curriculum weights
DEFAULT_CURRICULUM: list[dict] = [
    {
        "name": "open-web-math/open-web-math",
        "split": "train",
        "text_key": "text",
        "weight": 0.50,
        "label": "math",
    },
    {
        "name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_key": "text",
        "weight": 0.30,
        "label": "general",
    },
    {
        "name": "roneneldan/TinyStories",
        "split": "train",
        "text_key": "text",
        "weight": 0.20,
        "label": "narrative",
    },
]


class CurriculumDataLoader:
    """Streams, tokenizes, and chunks text from multiple datasets.

    Mixes data sources according to curriculum weights for balanced
    domain coverage during autoencoder and backbone training.

    Attributes:
        chunks: (N, K) int32 array of token chunks after loading
        chunk_labels: list of domain labels per chunk (for diagnostics)
        sources: curriculum source configurations
    """

    def __init__(
        self,
        tokenizer,
        chunk_size: int = 4,
        max_seq_length: int = 512,
        min_text_length: int = 50,
        curriculum: list[dict] | None = None,
    ) -> None:
        """Initialize the data loader.

        Args:
            tokenizer: HuggingFace tokenizer instance (e.g., Qwen3.5)
            chunk_size: K — tokens per chunk (must match CALM config)
            max_seq_length: max tokens per document before truncation
            min_text_length: skip documents shorter than this (chars)
            curriculum: list of dataset source dicts (default: math/general/narrative)
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_seq_length = max_seq_length
        self.min_text_length = min_text_length
        self.sources = curriculum or DEFAULT_CURRICULUM

        self.chunks: np.ndarray | None = None
        self.chunk_labels: list[str] = []
        self._domain_counts: dict[str, int] = {}

    def _tokenize_and_chunk(
        self,
        dataset_iter,
        text_key: str,
        target_chunks: int,
        label: str,
    ) -> list[np.ndarray]:
        """Tokenize documents from a streaming dataset into K-token chunks.

        Args:
            dataset_iter: iterable of dicts with text_key field
            text_key: key for text content in each example
            target_chunks: stop after collecting this many chunks
            label: domain label for tracking

        Returns:
            List of (M, K) chunk arrays from each document
        """
        chunk_buffer: list[np.ndarray] = []
        total = 0
        k = self.chunk_size

        for example in dataset_iter:
            text = example.get(text_key, "")
            if not text or len(text) < self.min_text_length:
                continue

            tokens = self.tokenizer.encode(
                text, max_length=self.max_seq_length, truncation=True
            )
            usable = len(tokens) - (len(tokens) % k)
            if usable < k:
                continue

            doc_chunks = np.array(tokens[:usable], dtype=np.int32).reshape(-1, k)
            chunk_buffer.append(doc_chunks)
            total += doc_chunks.shape[0]

            if total >= target_chunks:
                break

        self._domain_counts[label] = total
        return chunk_buffer

    def load(
        self,
        target_chunks: int = 250_000,
        verbose: bool = True,
    ) -> None:
        """Load and tokenize data from all curriculum sources.

        Distributes target_chunks across sources according to weights.
        Requires `datasets` and `transformers` packages.

        Args:
            target_chunks: total chunks to collect across all sources
            verbose: print progress during loading
        """
        from datasets import load_dataset  # noqa: E402

        all_buffers: list[np.ndarray] = []
        all_labels: list[str] = []

        for source in self.sources:
            source_target = int(target_chunks * source["weight"])
            label = source.get("label", source["name"])

            if verbose:
                print(
                    f"  Loading {label}: {source['name']} "
                    f"(target: {source_target:,} chunks)..."
                )

            try:
                load_kwargs = {
                    "path": source["name"],
                    "split": source.get("split", "train"),
                    "streaming": True,
                }
                if "config" in source:
                    load_kwargs["name"] = source["config"]

                ds = load_dataset(**load_kwargs)

                buffers = self._tokenize_and_chunk(
                    ds, source["text_key"], source_target, label
                )

                if buffers:
                    combined = np.concatenate(buffers, axis=0)[:source_target]
                    all_buffers.append(combined)
                    all_labels.extend([label] * combined.shape[0])

                    if verbose:
                        print(f"    ✓ {combined.shape[0]:,} chunks from {label}")
                else:
                    if verbose:
                        print(f"    ⚠ No data loaded from {label}")

            except Exception as exc:
                if verbose:
                    print(f"    ✗ Failed to load {label}: {exc}")
                    print("      Falling back to random tokens for this source")
                # fallback: random token chunks so training can still proceed
                vocab_size = len(self.tokenizer)
                fallback = np.random.randint(
                    0, vocab_size, size=(source_target, self.chunk_size), dtype=np.int32
                )
                all_buffers.append(fallback)
                all_labels.extend([f"{label}_fallback"] * source_target)

        if all_buffers:
            self.chunks = np.concatenate(all_buffers, axis=0)
            self.chunk_labels = all_labels
        else:
            raise RuntimeError("No data loaded from any source")

        if verbose:
            self._print_summary()

    def _print_summary(self) -> None:
        """Print dataset composition summary."""
        assert self.chunks is not None
        total = self.chunks.shape[0]
        k = self.chunk_size
        print("\n  Dataset summary:")
        print(f"    Total: {total:,} chunks × K={k} = {total * k:,} tokens")

        for label, count in self._domain_counts.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"    {label}: {count:,} ({pct:.1f}%)")

    def get_batch(
        self,
        key: jax.Array,
        batch_size: int = 64,
    ) -> jnp.ndarray:
        """Sample a random batch of chunks.

        Args:
            key: PRNG key for random sampling
            batch_size: number of chunks per batch

        Returns:
            (batch_size, K) jnp array of token IDs
        """
        assert self.chunks is not None, "Call .load() first"
        num_chunks = self.chunks.shape[0]
        indices = jax.random.randint(key, (batch_size,), 0, num_chunks)
        return jnp.array(self.chunks[indices])

    def get_task_batch(
        self,
        key: jax.Array,
        label: str,
        batch_size: int = 64,
    ) -> jnp.ndarray:
        """Sample a batch from a specific domain (for GEA task distribution).

        Args:
            key: PRNG key
            label: domain label (e.g., "math", "general", "narrative")
            batch_size: number of chunks

        Returns:
            (batch_size, K) jnp array of token IDs from that domain
        """
        assert self.chunks is not None, "Call .load() first"
        # find indices matching the requested label
        domain_indices = np.array(
            [i for i, lab in enumerate(self.chunk_labels) if lab == label],
            dtype=np.int32,
        )
        if len(domain_indices) == 0:
            # fallback to random chunks from any domain
            return self.get_batch(key, batch_size)

        # sample from domain-specific indices
        idx_positions = jax.random.randint(key, (batch_size,), 0, len(domain_indices))
        selected = domain_indices[np.array(idx_positions)]
        return jnp.array(self.chunks[selected])

    def get_task_distribution(self) -> list[dict]:
        """Build a task distribution for GEA evolution.

        Returns a list of task dicts, one per domain, each containing
        the domain label. Used with GroupEvolver.evaluate_population().

        Returns:
            List of {"type": label} dicts
        """
        labels = sorted(set(self.chunk_labels))
        return [{"type": label} for label in labels if not label.endswith("_fallback")]
