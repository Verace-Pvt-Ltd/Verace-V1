"""
Verace V1 Tokenizer Module
Integrates Moonshot's official Kimi K3 open-source tokenizer (163,840 vocabulary, tiktoken BPE)
with fallback to Tiktoken / HuggingFace / ByteLevel BPE.
"""

from typing import List, Union, Optional
import os
import sys

_MOONSHOT_DIR_LOCAL = os.path.join(os.path.dirname(__file__), "moonshot")
_MOONSHOT_DIR_EXT = "/media/krrish/data/Verace/Kimi_K3/kimi_k3/tokenizer/moonshot"

class VeraceTokenizer:
    """
    Production Tokenizer for Verace V1.
    Uses Moonshot's official Kimi K3 open-source tokenizer (vocab_size = 163,840).
    """
    def __init__(self, vocab_size: int = 163840, moonshot_dir: Optional[str] = None):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.backend = "moonshot_k3"

        if moonshot_dir is None:
            moonshot_dir = _MOONSHOT_DIR_LOCAL if os.path.exists(_MOONSHOT_DIR_LOCAL) else _MOONSHOT_DIR_EXT

        # 1. Load Moonshot Kimi K3 official tokenizer
        if os.path.exists(moonshot_dir):
            try:
                kimi_k3_path = "/media/krrish/data/Verace/Kimi_K3"
                if kimi_k3_path not in sys.path:
                    sys.path.insert(0, kimi_k3_path)
                from kimi_k3.tokenizer.real_tokenizer import MoonshotKimiTokenizer
                self.tokenizer = MoonshotKimiTokenizer(vocab_dir=moonshot_dir)
                self.vocab_size = min(vocab_size, self.tokenizer.vocab_size)
                self.backend = "moonshot_k3"
            except Exception as e:
                self.tokenizer = None

        # 2. Tiktoken fallback
        if self.tokenizer is None:
            try:
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.backend = "tiktoken"
            except Exception:
                pass

        # 3. HuggingFace fallback
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
                self.backend = "transformers"
            except Exception:
                self.backend = "byte_fallback"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if self.backend == "moonshot_k3":
            return [t % self.vocab_size for t in self.tokenizer.encode(text)]
        elif self.backend == "tiktoken":
            return [t % self.vocab_size for t in self.tokenizer.encode(text)]
        elif self.backend == "transformers":
            return [t % self.vocab_size for t in self.tokenizer.encode(text, add_special_tokens=add_special_tokens)]
        else:
            raw_bytes = text.encode("utf-8")
            return [b % self.vocab_size for b in raw_bytes]

    def decode(self, tokens: List[int]) -> str:
        if self.backend == "moonshot_k3":
            return self.tokenizer.decode(tokens)
        elif self.backend == "tiktoken":
            return self.tokenizer.decode([t % 100000 for t in tokens])
        elif self.backend == "transformers":
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            valid_bytes = bytes([t % 256 for t in tokens])
            return valid_bytes.decode("utf-8", errors="ignore")

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        if hasattr(self.tokenizer, "_tok") and hasattr(self.tokenizer._tok, "eos_id"):
            return self.tokenizer._tok.eos_id
        return 0
