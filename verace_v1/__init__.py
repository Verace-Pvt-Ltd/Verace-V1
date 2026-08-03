"""
Verace V1: A next-generation, multimodal large language model architecture.
Package initialization exposing SSSD attention, CHAM memory, M-CMoE, the ACD engine,
the vision encoder, the Unitary Muon optimizer, and the Hyper-XTML chat template.
"""

from verace_v1.config import VeraceV1Config
from verace_v1.modules.sssd_attention import SSSDAttention
from verace_v1.modules.cham_memory import ContinuousHolographicMemory
from verace_v1.modules.mcmoe import ManifoldContinuousMoE
from verace_v1.modules.acd_engine import AdaptiveCognitiveDepthEngine
from verace_v1.modules.energy_critic import LatentEnergyCritic
from verace_v1.modules.backbone import VeraceV1Model, VeraceV1Layer
from verace_v1.optimizer.unitary_muon import UnitaryMuon
from verace_v1.chat_template.hyper_xtml import HyperXTMLFormatter, HyperThought
from verace_v1.serving.hyper_generate import VeraceV1Generator
from verace_v1.eval.benchmark_runner import VeraceV1Evaluator

__all__ = [
    "VeraceV1Config",
    "SSSDAttention",
    "ContinuousHolographicMemory",
    "ManifoldContinuousMoE",
    "AdaptiveCognitiveDepthEngine",
    "LatentEnergyCritic",
    "VeraceV1Model",
    "VeraceV1Layer",
    "UnitaryMuon",
    "HyperXTMLFormatter",
    "HyperThought",
    "VeraceV1Generator",
    "VeraceV1Evaluator",
]
