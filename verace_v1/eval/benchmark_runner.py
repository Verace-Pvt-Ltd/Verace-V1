"""
Verace V1 Benchmark Evaluation Suite
Evaluates reasoning accuracy, FLOP reduction ratio via Adaptive Cognitive Depth,
and cognitive depth unrolling distribution.
"""

from typing import Dict, Any, List, Optional
import torch
from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.serving.hyper_generate import VeraceV1Generator

class VeraceV1Evaluator:
    """Evaluator for Verace V1 model capabilities."""
    def __init__(self, model: VeraceV1Model, config: VeraceV1Config):
        self.model = model
        self.config = config
        self.generator = VeraceV1Generator(model, config)

    def evaluate_benchmark(
        self,
        benchmark_name: str,
        test_samples: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        if test_samples is None:
            test_samples = [
                {"prompt": "Prove the Riemann Hypothesis analogue for finite fields", "ground_truth": "proved"},
                {"prompt": "Synthesize a quantum error correcting code", "ground_truth": "surface code"}
            ]

        correct_count = 0
        total_count = len(test_samples)
        depth_sum = 0.0

        for sample in test_samples:
            prompt = sample["prompt"]
            truth = sample["ground_truth"]

            output = self.generator.generate(prompt_text=prompt, max_new_tokens=64)
            if truth.lower() in output.lower():
                correct_count += 1

            depth_sum += self._measure_cognitive_depth(prompt + output)

        accuracy = (float(correct_count) / float(total_count)) * 100.0
        avg_depth = depth_sum / total_count

        return {
            "benchmark": benchmark_name,
            "accuracy_score": accuracy,
            "avg_cognitive_depth": avg_depth,
            "flop_reduction_factor": f"{self.config.num_layers / avg_depth:.1f}x"
        }

    def _measure_cognitive_depth(self, text: str) -> float:
        """
        Runs a forward pass over `text` and returns the mean number of layers
        actually executed per token (see AdaptiveCognitiveDepthEngine), rather
        than an assumed constant.
        """
        input_bytes = text.encode("utf-8")
        input_ids = torch.tensor(
            [[b % self.config.vocab_size for b in input_bytes]],
            dtype=torch.long,
            device=next(self.model.parameters()).device
        )
        with torch.no_grad():
            _, depth_counts = self.model(input_ids, use_adaptive_depth=True)
        return depth_counts.float().mean().item()
