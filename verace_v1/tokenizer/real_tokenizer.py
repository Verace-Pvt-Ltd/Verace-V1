"""
Wrapper around Moonshot's actual released Kimi K3 tokenizer (vendored, unmodified, in
verace_v1/tokenizer/moonshot/ -- see NOTICE there for provenance/license). This is the
real tiktoken-based BPE + XTML chat-template renderer Kimi K3 uses, not a re-derivation
from the paper's prose.
"""

import json
import os
from typing import List, Optional

from verace_v1.tokenizer.moonshot.tokenization_kimi import TikTokenTokenizer
from verace_v1.tokenizer.moonshot.encoding_k3 import build_chat_segments

_MOONSHOT_DIR = os.path.join(os.path.dirname(__file__), "moonshot")


class _AddedTokenView:
    """Minimal shim: TikTokenTokenizer only reads `.content` off added_tokens_decoder values."""
    def __init__(self, content: str):
        self.content = content


class MoonshotKimiTokenizer:
    """
    The real Kimi K3 tokenizer (vocab_size=163,840 -- the paper's Table 1 "160K" rounds this).
    """
    def __init__(self, vocab_dir: str = _MOONSHOT_DIR):
        cfg_path = os.path.join(vocab_dir, "tokenizer_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        added_tokens_decoder = {
            int(k): _AddedTokenView(v["content"]) for k, v in cfg["added_tokens_decoder"].items()
        }
        self._tok = TikTokenTokenizer(
            vocab_file=os.path.join(vocab_dir, "tiktoken.model"),
            bos_token=cfg["bos_token"],
            eos_token=cfg["eos_token"],
            unk_token=cfg.get("unk_token"),
            pad_token=cfg.get("pad_token"),
            additional_special_tokens=cfg.get("additional_special_tokens"),
            added_tokens_decoder=added_tokens_decoder,
        )

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._tok.decode(token_ids)

    def token_to_id(self, token: str) -> Optional[int]:
        tid = self._tok._convert_token_to_id(token)
        return tid if tid != self._tok.unk_id or token == self._tok.unk_token else tid

    def render_chat(
        self,
        messages: list,
        tools: Optional[list] = None,
        add_generation_prompt: bool = True,
        thinking: bool = True,
        thinking_effort: str = "max"
    ) -> str:
        """
        Renders `messages` (list of {"role": ..., "content": ...} dicts, HF chat-template
        convention) into the real XTML text stream via Moonshot's own `build_chat_segments` --
        the actual Appendix F implementation, not a re-derivation.
        """
        segments = build_chat_segments(
            messages, tools=tools, add_generation_prompt=add_generation_prompt,
            thinking=thinking, thinking_effort=thinking_effort,
        )
        return "".join(segment.text for segment in segments)

    def render_and_encode_chat(self, messages: list, **kwargs) -> List[int]:
        text = self.render_chat(messages, **kwargs)
        return self.encode(text)
