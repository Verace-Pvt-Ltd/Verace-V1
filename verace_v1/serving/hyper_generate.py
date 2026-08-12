"""
Latent Tree Search Generation Engine for Verace V1
Autoregressive decoding through VeraceV1Model with adaptive-depth compute per generated
token. When tree search is enabled, each step samples top-K candidate next tokens,
scores each candidate's resulting latent state against the context with
LatentEnergyCritic, and commits the minimum-energy candidate rather than a single
temperature sample.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.chat_template.hyper_xtml import HyperXTMLFormatter, HyperThought

class VeraceV1Generator:
    """
    Verace V1 generator: adaptive-depth decoding with optional latent tree search.
    """
    def __init__(self, model: VeraceV1Model, config: VeraceV1Config, tokenizer: Optional[Any] = None):
        self.model = model
        self.config = config
        self.formatter = HyperXTMLFormatter()
        if tokenizer is None:
            from verace_v1.tokenizer import VeraceTokenizer
            self.tokenizer = VeraceTokenizer(vocab_size=config.vocab_size)
        else:
            self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        use_tree_search: bool = True,
        num_branches: Optional[int] = None
    ) -> str:
        """
        use_tree_search: if True (default), select each token by minimum latent energy
            over `num_branches` sampled candidates.
        """
        self.model.eval()
        num_branches = num_branches or self.config.tree_branches

        tokens = self.tokenizer.encode(prompt_text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=next(self.model.parameters()).device)

        generated_tokens = []

        for _ in range(max_new_tokens):
            logits, depth_counts, hidden = self.model(input_ids, use_adaptive_depth=True, return_hidden=True)

            next_logits = logits[0, -1, :] / max(1e-5, temperature)
            probs = F.softmax(next_logits, dim=-1)

            if use_tree_search and num_branches > 1:
                next_token = self._select_branch_by_energy(input_ids, probs, hidden, num_branches)
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=input_ids.device)], dim=1)

            if next_token == self.tokenizer.eos_token_id and len(generated_tokens) > 1:
                break

        completion = self.tokenizer.decode(generated_tokens)
        return completion

    def _select_branch_by_energy(
        self,
        input_ids: torch.Tensor,
        probs: torch.Tensor,
        context_hidden: torch.Tensor,
        num_branches: int
    ) -> int:
        """
        Samples `num_branches` candidate next tokens from `probs` (without replacement),
        runs the model forward on each resulting sequence to get its latent state, and
        returns the token whose resulting state has minimum energy against the current
        context (LatentEnergyCritic.compute_energy) — see docs/modules/energy-critic.md.

        Costs num_branches additional full forward passes per generated token.
        """
        num_branches = min(num_branches, int((probs > 0).sum().item()))
        topk_tokens = torch.multinomial(probs, num_samples=num_branches, replacement=False)

        x_prompt = context_hidden[:, -1:, :]  # [1, 1, hidden_dim]

        candidate_hiddens = []
        for tok in topk_tokens.tolist():
            cand_ids = torch.cat(
                [input_ids, torch.tensor([[tok]], device=input_ids.device)], dim=1
            )
            _, _, cand_hidden = self.model(cand_ids, use_adaptive_depth=True, return_hidden=True)
            candidate_hiddens.append(cand_hidden[:, -1:, :])

        h_cand = torch.stack(candidate_hiddens, dim=1)  # [1, num_branches, 1, hidden_dim]
        energies = self.model.energy_critic.compute_energy(x_prompt, h_cand)  # [1, num_branches]
        best_idx = torch.argmin(energies, dim=-1).item()

        return topk_tokens[best_idx].item()
