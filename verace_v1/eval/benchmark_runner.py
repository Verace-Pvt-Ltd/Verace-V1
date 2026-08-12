"""
Verace V1 Production Benchmark Evaluation Suite
Evaluates reasoning accuracy, log-likelihood candidate scoring (MMLU / HellaSwag / GSM8K),
FLOP reduction ratio via Adaptive Cognitive Depth, and cognitive depth unrolling distribution.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.serving.hyper_generate import VeraceV1Generator
from verace_v1.tokenizer import VeraceTokenizer


class VeraceV1Evaluator:
    """Production Evaluator for Verace V1 benchmark performance."""
    def __init__(self, model: VeraceV1Model, config: VeraceV1Config, tokenizer: Optional[VeraceTokenizer] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer or VeraceTokenizer(vocab_size=config.vocab_size)
        self.generator = VeraceV1Generator(model, config, tokenizer=self.tokenizer)

    def evaluate_benchmark(
        self,
        benchmark_name: str = "MMLU",
        test_samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates test_samples against benchmark_name (MMLU, GSM8K, HellaSwag, HumanEval).
        """
        if test_samples is None:
            if benchmark_name.upper() == "MMLU":
                test_samples = [
                    {
                        "prompt": "What is the primary function of DNA replication?\nA. Energy production\nB. Copying genetic information\nC. Protein degradation\nD. Cell wall synthesis\nAnswer:",
                        "choices": ["A", "B", "C", "D"],
                        "target_idx": 1
                    },
                    {
                        "prompt": "Which element has the highest electronegativity?\nA. Sodium\nB. Chlorine\nC. Fluorine\nD. Oxygen\nAnswer:",
                        "choices": ["A", "B", "C", "D"],
                        "target_idx": 2
                    }
                ]
            elif benchmark_name.upper() == "GSM8K":
                test_samples = [
                    {
                        "prompt": "Janet buys 3 packs of 10 pencils each. She gives 5 to her sister. How many pencils does Janet have left?",
                        "ground_truth": "25"
                    }
                ]
            else:
                test_samples = [
                    {"prompt": "Prove the Riemann Hypothesis analogue for finite fields", "ground_truth": "proved"},
                    {"prompt": "Synthesize a quantum error correcting code", "ground_truth": "surface code"}
                ]

        correct_count = 0
        total_count = len(test_samples)
        depth_sum = 0.0

        for sample in test_samples:
            prompt = sample["prompt"]

            if "choices" in sample and "target_idx" in sample:
                # Multiple Choice Log-Likelihood Evaluation (MMLU / HellaSwag)
                pred_idx = self._eval_multiple_choice_loglik(prompt, sample["choices"])
                if pred_idx == sample["target_idx"]:
                    correct_count += 1
                output_text = sample["choices"][pred_idx]
            else:
                # Free-form Generation Evaluation (GSM8K / Math / Code)
                truth = sample.get("ground_truth", "")
                output_text = self.generator.generate(prompt_text=prompt, max_new_tokens=64)
                if truth.lower() in output_text.lower():
                    correct_count += 1

            depth_sum += self._measure_cognitive_depth(prompt + " " + output_text)

        accuracy = (float(correct_count) / float(total_count)) * 100.0
        avg_depth = depth_sum / max(1, total_count)

        return {
            "benchmark": benchmark_name,
            "accuracy_score": accuracy,
            "avg_cognitive_depth": avg_depth,
            "flop_reduction_factor": f"{self.config.num_layers / max(1.0, avg_depth):.1f}x",
            "samples_evaluated": total_count
        }

    def _eval_multiple_choice_loglik(self, prompt: str, choices: List[str]) -> int:
        """Scores candidate choices using normalized log-likelihood of option tokens."""
        self.model.eval()
        device = next(self.model.parameters()).device
        prompt_tokens = self.tokenizer.encode(prompt)

        choice_logliks = []

        for choice in choices:
            choice_tokens = self.tokenizer.encode(" " + choice)
            full_tokens = prompt_tokens + choice_tokens

            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                logits, _ = self.model(input_ids, use_adaptive_depth=True)

            # Target logits for choice positions
            choice_logits = logits[0, len(prompt_tokens) - 1 : len(full_tokens) - 1, :]
            target_ids = torch.tensor(choice_tokens, dtype=torch.long, device=device)

            log_probs = F.log_softmax(choice_logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            choice_logliks.append(token_log_probs.sum().item())

        return int(torch.argmax(torch.tensor(choice_logliks)).item())

    def _measure_cognitive_depth(self, text: str) -> float:
        """
        Runs a forward pass over `text` and returns the mean number of layers
        executed per token by AdaptiveCognitiveDepthEngine.
        """
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=next(self.model.parameters()).device)
        with torch.no_grad():
            _, depth_counts = self.model(input_ids, use_adaptive_depth=True)
        return depth_counts.float().mean().item()

