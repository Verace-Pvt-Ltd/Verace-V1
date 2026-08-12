"""
Verace V1 Dataset and DataLoader Module
Implements PyTorch Dataset and DataLoader pipelines for text pre-training with token packing.
"""

import array
import json
import os
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

from verace_v1.tokenizer import VeraceTokenizer

# Bytes of raw text accumulated before each tokenizer.encode() call while streaming a
# file -- bounds peak memory (tokenizing the whole file in one call previously OOM'd on
# real-sized corpora: a 2.2GB text file tokenizes to ~500M+ tokens, which as a plain
# Python list of ints is 15GB+ due to per-object overhead) without so many tiny calls
# that per-call overhead dominates.
_TOKENIZE_CHUNK_BYTES = 4 * 1024 * 1024

class TextDataset(Dataset):
    """
    Token-packed PyTorch Dataset for Verace V1 pre-training.
    Tokenizes raw text or JSONL files into sequence blocks of fixed context_length.
    Streams input files in bounded chunks and stores tokens in an array.array('i')
    (4 bytes/token, no per-element Python object overhead) rather than a plain list,
    so real-sized corpora (hundreds of MB to low GB of text) don't exhaust RAM.
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

        self.samples = []
        self._load_and_tokenize(data_path)

    def _encode_chunks(self, line_iter):
        """Yields *batches* (lists) of token ids from an iterator of lines, tokenizing in
        ~_TOKENIZE_CHUNK_BYTES batches (bounded per-call memory, and yielding per-chunk
        rather than per-token keeps this fast -- array.extend(chunk) instead of hundreds
        of millions of individual array.append() calls)."""
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

    def _load_and_tokenize(self, data_path: Optional[str]):
        token_stream = array.array("i")

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
            token_stream.extend(self.tokenizer.encode(raw_text))
            files = []

        for fpath in files:
            if self.max_tokens is not None and len(token_stream) >= self.max_tokens:
                break
            if fpath.endswith(".jsonl"):
                def _jsonl_texts(fp=fpath):
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                doc = json.loads(line)
                                yield doc.get("text", doc.get("content", "")) + "\n"
                for chunk in self._encode_chunks(_jsonl_texts()):
                    token_stream.extend(chunk)
                    if self.max_tokens is not None and len(token_stream) >= self.max_tokens:
                        break
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    for chunk in self._encode_chunks(f):
                        token_stream.extend(chunk)
                        if self.max_tokens is not None and len(token_stream) >= self.max_tokens:
                            break

        if self.max_tokens is not None and len(token_stream) > self.max_tokens:
            token_stream = token_stream[:self.max_tokens]

        # Pack tokens into fixed sequence lengths
        num_tokens = len(token_stream)
        for i in range(0, num_tokens - self.context_length, self.stride):
            seq = token_stream[i : i + self.context_length + 1]
            if len(seq) == self.context_length + 1:
                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                labels = torch.tensor(seq[1:], dtype=torch.long)
                self.samples.append({"input_ids": input_ids, "labels": labels})

        if len(self.samples) == 0:
            # Fallback: repeat the (too-short) real token stream to fill one context window.
            warnings.warn(
                f"TextDataset: '{data_path}' produced only {num_tokens} tokens, fewer than "
                f"context_length={self.context_length}. Repeating them to fill one sample -- "
                f"provide more data for a real training run.",
                stacklevel=2
            )
            reps = (self.context_length + 2) // max(1, len(token_stream)) + 1
            seq = array.array("i", token_stream.tolist() * reps)[: self.context_length + 1]
            input_ids = torch.tensor(seq[:-1], dtype=torch.long)
            labels = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


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
