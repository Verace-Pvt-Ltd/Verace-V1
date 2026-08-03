"""
Hyper-XTML Multidimensional Thought Template Module
Supports multidimensional reasoning channels, energy bounds, and tree branch rollouts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import re

OPEN_TOKEN = "[open]"
SEP_TOKEN = "[sep]"
CLOSE_TOKEN = "[close]"
END_OF_MSG_TOKEN = "[end_of_msg]"

@dataclass
class HyperThought:
    content: str
    cognitive_depth: int
    energy_score: float
    branch_id: int = 0


class HyperXTMLFormatter:
    """Formatter and Parser for Verace V1 Hyper-XTML Multidimensional Thoughts"""
    def __init__(self):
        self.open_t = OPEN_TOKEN
        self.sep_t = SEP_TOKEN
        self.close_t = CLOSE_TOKEN
        self.end_t = END_OF_MSG_TOKEN

    def format_hyper_thought(self, thought: HyperThought) -> str:
        return (
            f"{self.open_t}hyper_think depth=\"{thought.cognitive_depth}\" "
            f"energy=\"{thought.energy_score:.4f}\" branch=\"{thought.branch_id}\"{self.sep_t}\n"
            f"{thought.content}\n"
            f"{self.close_t}hyper_think{self.sep_t}"
        )

    def parse_hyper_thought(self, text: str) -> Optional[HyperThought]:
        match = re.search(
            r"\[open\]hyper_think depth=\"(\d+)\" energy=\"([\d\.]+)\" branch=\"(\d+)\"\[sep\](.*?)\[close\]hyper_think\[sep\]",
            text, re.DOTALL
        )
        if match:
            depth = int(match.group(1))
            energy = float(match.group(2))
            branch = int(match.group(3))
            content = match.group(4).strip()
            return HyperThought(content=content, cognitive_depth=depth, energy_score=energy, branch_id=branch)
        return None
