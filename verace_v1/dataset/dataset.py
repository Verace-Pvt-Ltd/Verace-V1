"""
Verace V1 Dataset and DataLoader Module
Implements PyTorch Dataset and DataLoader pipelines for text pre-training with token packing.

Tokenizes a corpus once to an on-disk binary token cache (int32, raw), then serves
training windows via numpy.memmap -- RAM usage for the token data itself stays O(1)
regardless of corpus size (an earlier in-RAM design, even after switching from a plain
Python list to array.array, still OOM-killed on a real ~2.2GB corpus: holding the
*entire* tokenized stream as one process-resident object doesn't scale no matter which
container holds it. Memory-mapping a file backing it is the standard fix real training
pipelines use, e.g. nanoGPT's data/prepare.py).
"""

import hashlib
import json
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional

from verace_v1.tokenizer import VeraceTokenizer

# Bytes of raw text accumulated before each tokenizer.encode() call while streaming a
# file -- bounds peak memory per call without so many tiny calls that per-call overhead
# (Python/HF tokenizer call overhead) dominates.
_TOKENIZE_CHUNK_BYTES = 4 * 1024 * 1024


def resolve_corpus_files(data_path: Optional[str]) -> List[str]:
    """Resolves data_path (a single .txt/.jsonl file, or a directory of them) to a sorted
    list of file paths. Returns [] if data_path is None or doesn't exist (callers should
    fall back to synthetic data in that case, as TextDataset itself does)."""
    if data_path is not None and os.path.isfile(data_path):
        return [data_path]
    if data_path is not None and os.path.isdir(data_path):
        return sorted(
            os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.txt', '.jsonl'))
        )
    return []

class TextDataset(Dataset):
    """
    Token-packed PyTorch Dataset for Verace V1 pre-training.
    Tokenizes raw text or JSONL files into sequence blocks of fixed context_length.

    Real corpora are tokenized once to an on-disk int32 binary cache (streamed in
    ~_TOKENIZE_CHUNK_BYTES batches, never holding the full token stream in RAM), keyed
    by (source file paths+mtimes+sizes, tokenizer backend/vocab, max_tokens) so a repeat
    run against the same corpus/tokenizer reuses the cache instead of re-tokenizing.
    Training windows are then served via numpy.memmap: __getitem__ pages in only the
    ~context_length tokens it needs, so RAM usage for the token data is O(1) in corpus
    size, not O(corpus size).
    """
    def __init__(
        self,
        data_path: Optional[str],
        tokenizer: Optional[VeraceTokenizer] = None,
        context_length: int = 4096,
        stride: Optional[int] = None,
        max_tokens: Optional[int] = None
    ):
        self.context_length = context_length
        self.tokenizer = tokenizer or VeraceTokenizer()
        self.stride = stride or context_length
        self.max_tokens = max_tokens

        self._token_source: Optional[np.ndarray] = None  # memmap (real corpus) or small in-RAM array (fallbacks)
        self.window_starts: range = range(0)
        self._load_and_tokenize(data_path)

    def _encode_chunks(self, line_iter):
        """Yields *batches* (lists) of token ids from an iterator of lines, tokenizing in
        ~_TOKENIZE_CHUNK_BYTES batches (bounded per-call memory, and yielding per-chunk
        rather than per-token keeps this fast)."""
        buf: List[str] = []
        buf_bytes = 0
        for line in line_iter:
            buf.append(line)
            buf_bytes += len(line.encode("utf-8"))
            if buf_bytes >= _TOKENIZE_CHUNK_BYTES:
                yield self.tokenizer.encode("".join(buf))
                buf, buf_bytes = [], 0
        if buf:
            yield self.tokenizer.encode("".join(buf))

    def _jsonl_lines(self, fpath: str):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    yield doc.get("text", doc.get("content", "")) + "\n"

    def _cache_path_for(self, files: List[str]) -> str:
        """Deterministic on-disk cache path for (files+mtimes+sizes, tokenizer, max_tokens)
        -- changes to the source files or tokenizer automatically invalidate the cache."""
        key_parts = [
            f"{os.path.abspath(f)}:{os.path.getmtime(f)}:{os.path.getsize(f)}" for f in sorted(files)
        ]
        key = "|".join(key_parts) + f"|backend={self.tokenizer.backend}|vocab={self.tokenizer.vocab_size}|max_tokens={self.max_tokens}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(files[0])), ".verace_token_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"tokens_{digest}.bin")

    def _tokenize_to_binary(self, files: List[str], out_path: str) -> int:
        """Streams `files`, tokenizes in bounded chunks, and appends each chunk's tokens
        (as raw int32 bytes) directly to disk -- the full token stream is never resident
        in RAM at once. Writes to a .tmp path and renames atomically on success, so a
        killed/interrupted run never leaves a corrupt cache file behind."""
        tmp_path = out_path + ".tmp"
        written = 0
        with open(tmp_path, "wb") as out:
            for fpath in files:
                if self.max_tokens is not None and written >= self.max_tokens:
                    break
                line_iter = self._jsonl_lines(fpath) if fpath.endswith(".jsonl") else open(fpath, "r", encoding="utf-8")
                try:
                    for chunk in self._encode_chunks(line_iter):
                        if self.max_tokens is not None:
                            remaining = self.max_tokens - written
                            if remaining <= 0:
                                break
                            chunk = chunk[:remaining]
                        np.asarray(chunk, dtype=np.int32).tofile(out)
                        written += len(chunk)
                        if self.max_tokens is not None and written >= self.max_tokens:
                            break
                finally:
                    if hasattr(line_iter, "close"):
                        line_iter.close()
        os.replace(tmp_path, out_path)
        return written

    def _load_and_tokenize(self, data_path: Optional[str]):
        if data_path is not None and os.path.isfile(data_path):
            files = [data_path]
        elif data_path is not None and os.path.isdir(data_path):
            files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.txt', '.jsonl'))]
        else:
            # Fallback synthetic data if file does not exist -- loud on purpose: training
            # on this instead of a real corpus by accident should never pass silently.
            warnings.warn(
                f"TextDataset: data_path '{data_path}' does not exist as a file or "
                f"directory. Falling back to a repeated synthetic sentence -- any "
                f"training run using this dataset is NOT learning from real data.",
                stacklevel=2
            )
            raw_text = "Verace V1 Next Generation Intelligence Architecture with Spectral Attention and Continuous Associative Memory. " * 50
            self._token_source = np.asarray(self.tokenizer.encode(raw_text), dtype=np.int32)
            self._finalize_windows(data_path)
            return

        cache_path = self._cache_path_for(files)
        if not os.path.exists(cache_path):
            self._tokenize_to_binary(files, cache_path)

        self._token_source = np.memmap(cache_path, dtype=np.int32, mode="r")
        self._finalize_windows(data_path)

    def _finalize_windows(self, data_path: Optional[str]):
        num_tokens = len(self._token_source)
        # Only offsets that actually yield a full context_length+1 window.
        last_valid_start = num_tokens - self.context_length - 1
        if last_valid_start < 0:
            self.window_starts = range(0)
        else:
            self.window_starts = range(0, last_valid_start + 1, self.stride)

        if len(self.window_starts) == 0:
            # Fallback: repeat the (too-short) real token stream to fill one context window.
            warnings.warn(
                f"TextDataset: '{data_path}' produced only {num_tokens} tokens, fewer than "
                f"context_length={self.context_length}. Repeating them to fill one sample -- "
                f"provide more data for a real training run.",
                stacklevel=2
            )
            base = np.asarray(self._token_source)  # materialize (it's short by construction)
            reps = (self.context_length + 2) // max(1, len(base)) + 1
            self._token_source = np.tile(base, reps)[: self.context_length + 1]
            self.window_starts = range(0, 1)

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.window_starts[idx]
        seq = self._token_source[start: start + self.context_length + 1]
        input_ids = torch.tensor(seq[:-1].astype(np.int64), dtype=torch.long)
        labels = torch.tensor(seq[1:].astype(np.int64), dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def create_pretrain_dataloader(
    data_path: Optional[str],
    tokenizer: Optional[VeraceTokenizer] = None,
    batch_size: int = 4,
    context_length: int = 4096,
    num_workers: int = 2,
    shuffle: bool = True,
    max_tokens: Optional[int] = None
) -> DataLoader:
    """Helper to instantiate PyTorch DataLoader for pretraining."""
    dataset = TextDataset(data_path=data_path, tokenizer=tokenizer, context_length=context_length, max_tokens=max_tokens)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
