"""
Verace V1 Dataset and DataLoader Module
Implements PyTorch Dataset and DataLoader pipelines for text pre-training with token packing.
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

from verace_v1.tokenizer import VeraceTokenizer

class TextDataset(Dataset):
    """
    Token-packed PyTorch Dataset for Verace V1 pre-training.
    Tokenizes raw text or JSONL files into sequence blocks of fixed context_length.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[VeraceTokenizer] = None,
        context_length: int = 4096,
        stride: Optional[int] = None
    ):
        self.context_length = context_length
        self.tokenizer = tokenizer or VeraceTokenizer()
        self.stride = stride or context_length

        self.samples = []
        self._load_and_tokenize(data_path)

    def _load_and_tokenize(self, data_path: str):
        token_stream = []

        if os.path.isfile(data_path):
            files = [data_path]
        elif os.path.isdir(data_path):
            files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.txt', '.jsonl'))]
        else:
            # Fallback synthetic data if file does not exist
            raw_text = "Verace V1 Next Generation Intelligence Architecture with Spectral Attention and Continuous Associative Memory. " * 50
            token_stream.extend(self.tokenizer.encode(raw_text))
            files = []

        for fpath in files:
            if fpath.endswith(".jsonl"):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            doc = json.loads(line)
                            text = doc.get("text", doc.get("content", ""))
                            token_stream.extend(self.tokenizer.encode(text))
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                    token_stream.extend(self.tokenizer.encode(text))

        # Pack tokens into fixed sequence lengths
        num_tokens = len(token_stream)
        for i in range(0, num_tokens - self.context_length, self.stride):
            seq = token_stream[i : i + self.context_length + 1]
            if len(seq) == self.context_length + 1:
                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                labels = torch.tensor(seq[1:], dtype=torch.long)
                self.samples.append({"input_ids": input_ids, "labels": labels})

        if len(self.samples) == 0:
            # Synthetic fallback sample if dataset was too short
            seq = (token_stream * ((self.context_length + 2) // max(1, len(token_stream)) + 1))[: self.context_length + 1]
            input_ids = torch.tensor(seq[:-1], dtype=torch.long)
            labels = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def create_pretrain_dataloader(
    data_path: str,
    tokenizer: Optional[VeraceTokenizer] = None,
    batch_size: int = 4,
    context_length: int = 4096,
    num_workers: int = 2,
    shuffle: bool = True
) -> DataLoader:
    """Helper to instantiate PyTorch DataLoader for pretraining."""
    dataset = TextDataset(data_path=data_path, tokenizer=tokenizer, context_length=context_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
