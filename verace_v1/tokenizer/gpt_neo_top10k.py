"""
GPT-Neo tokenizer restricted to its 10,000 most frequent tokens -- matches the TinyStories
paper (Eldan & Li, arXiv:2305.07759) exactly: "We use GPT-Neo tokenizer but only keep the
top 10K most common tokens." GPT-Neo uses the standard GPT-2 BPE tokenizer/vocab (50,257
tokens); "most common" is determined by frequency in the actual training corpus, computed
once via a streaming scan and cached to disk.

Used specifically for apples-to-apples comparison against the paper's own published
numbers -- NOT the tokenizer Verace V1 uses elsewhere (see verace_v1/tokenizer/tokenizer.py,
the vendored Moonshot Kimi K3 tokenizer), since token-level loss/perplexity is only
comparable across runs using the identical tokenizer and vocabulary.
"""

import hashlib
import json
import os
from collections import Counter
from typing import List, Optional

VOCAB_SIZE = 10000
_TOKENIZE_CHUNK_BYTES = 4 * 1024 * 1024


class GPTNeoTop10KTokenizer:
    """
    vocab_size = 10000 (compressed ids 0..9999). Token id 0 is the OOV/unknown bucket for
    any GPT-2 BPE token outside the corpus's 10,000 most frequent tokens (this includes
    GPT-2's own byte-fallback tokens, so no input text is ever unencodable -- rare tokens
    just collapse to <unk>, the same lossy-but-total behavior "top 10K" implies).
    """
    UNK_ID = 0

    def __init__(self, rank_map_path: str):
        from transformers import AutoTokenizer
        self._base = AutoTokenizer.from_pretrained("gpt2")
        self.vocab_size = VOCAB_SIZE
        self.backend = "gpt_neo_top10k"

        with open(rank_map_path, "r") as f:
            data = json.load(f)
        # original GPT-2 token id (str key, JSON) -> compressed id (1..9999, 0 reserved for UNK)
        self._orig_to_compressed = {int(k): v for k, v in data["orig_to_compressed"].items()}
        self._compressed_to_orig = {v: int(k) for k, v in data["orig_to_compressed"].items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        orig_ids = self._base.encode(text)
        return [self._orig_to_compressed.get(t, self.UNK_ID) for t in orig_ids]

    def decode(self, tokens: List[int]) -> str:
        orig_ids = [self._compressed_to_orig.get(t, self._compressed_to_orig.get(self.UNK_ID, 0)) for t in tokens]
        return self._base.decode(orig_ids)

    @property
    def pad_token_id(self) -> int:
        return self.UNK_ID

    @property
    def eos_token_id(self) -> int:
        eot = self._base.eos_token_id  # GPT-2's <|endoftext|>, id 50256
        return self._orig_to_compressed.get(eot, self.UNK_ID)


def _iter_text_chunks(files: List[str], max_bytes: Optional[int] = None):
    """Streams raw text in ~_TOKENIZE_CHUNK_BYTES batches, capped at max_bytes total if given
    (frequency ranking from a large sample of the corpus is representative -- doesn't need
    the whole multi-GB file, and this keeps the one-time ranking pass fast)."""
    read_bytes = 0
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            buf, buf_bytes = [], 0
            for line in f:
                buf.append(line)
                nbytes = len(line.encode("utf-8"))
                buf_bytes += nbytes
                read_bytes += nbytes
                if buf_bytes >= _TOKENIZE_CHUNK_BYTES:
                    yield "".join(buf)
                    buf, buf_bytes = [], 0
                if max_bytes is not None and read_bytes >= max_bytes:
                    if buf:
                        yield "".join(buf)
                    return
            if buf:
                yield "".join(buf)


def build_or_load_rank_map(
    files: List[str],
    cache_dir: str,
    freq_sample_bytes: int = 500 * 1024 * 1024
) -> str:
    """
    Computes (or loads a cached) mapping of the corpus's 10,000 most frequent GPT-2 BPE
    token ids -> compressed ids 1..9999 (0 reserved for UNK), by frequency-scanning up to
    `freq_sample_bytes` of the corpus (default 500MB -- large enough to be representative,
    far cheaper than scanning a multi-GB corpus twice). Returns the path to the cached
    rank-map JSON file.
    """
    key_parts = [f"{os.path.abspath(f)}:{os.path.getmtime(f)}:{os.path.getsize(f)}" for f in sorted(files)]
    key = "|".join(key_parts) + f"|freq_sample_bytes={freq_sample_bytes}|vocab={VOCAB_SIZE}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    rank_map_path = os.path.join(cache_dir, f"gpt_neo_top10k_rankmap_{digest}.json")

    if os.path.exists(rank_map_path):
        return rank_map_path

    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained("gpt2")

    counts: Counter = Counter()
    for chunk_text in _iter_text_chunks(files, max_bytes=freq_sample_bytes):
        counts.update(base.encode(chunk_text))

    most_common = [tok_id for tok_id, _ in counts.most_common(VOCAB_SIZE - 1)]  # -1 for UNK slot
    orig_to_compressed = {tok_id: i + 1 for i, tok_id in enumerate(most_common)}  # compressed ids 1..9999

    tmp_path = rank_map_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"orig_to_compressed": orig_to_compressed}, f)
    os.replace(tmp_path, rank_map_path)
    return rank_map_path


def build_gpt_neo_top10k_tokenizer(files: List[str], cache_dir: str) -> GPTNeoTop10KTokenizer:
    """Convenience constructor: builds/loads the rank map for `files` and returns a ready
    GPTNeoTop10KTokenizer."""
    rank_map_path = build_or_load_rank_map(files, cache_dir)
    return GPTNeoTop10KTokenizer(rank_map_path)
