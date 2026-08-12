"""
End-to-End Integration Test for the Verace V1 Architecture
Tests Verace V1 model creation, SSSD attention, CHAM memory, M-CMoE, the ACD engine,
the Unitary Muon optimizer, a pretraining step, and generation.
"""

import torch
from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.optimizer.unitary_muon import UnitaryMuon
from verace_v1.training.pretrain import train_pretrain_step
from verace_v1.serving.hyper_generate import VeraceV1Generator
from verace_v1.eval.benchmark_runner import VeraceV1Evaluator

def test_end2end_pipeline():
    config = VeraceV1Config(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=2,
        head_dim=64,
        spectral_dim=32,
        chams_holographic_dim=32,
        mcmoe_rank=8,
        mcmoe_num_components=4,
        max_cognitive_depth=4,
        min_cognitive_depth=1
    )
    
    # 1. Model Instantiation
    model = VeraceV1Model(config)
    
    # 2. Input Tokens & Forward Pass
    b, s = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (b, s))
    labels = input_ids.clone()
    
    logits, depth_counts = model(input_ids, use_adaptive_depth=True)
    assert logits.shape == (b, s, config.vocab_size)
    assert depth_counts.shape == (b, s)
    assert not torch.isnan(logits).any()
    print(f"Forward pass successful. Average depth: {depth_counts.float().mean():.2f}")

    # 3. Unitary Muon Optimizer & Training Step
    optimizer = UnitaryMuon(model.parameters(), lr=0.01)
    batch = {"input_ids": input_ids, "labels": labels}
    
    ce_loss, mean_depth = train_pretrain_step(model, optimizer, batch)
    assert ce_loss > 0.0
    print(f"Verace V1 pretraining loss: {ce_loss:.4f}, mean cognitive depth: {mean_depth:.2f}")

    # 4. Generation Test
    generator = VeraceV1Generator(model, config)
    completion = generator.generate("Describe the Verace V1 architecture", max_new_tokens=8, use_tree_search=False)
    assert len(completion) > 0
    print("Generation test passed.")

    # 5. Evaluator Test
    evaluator = VeraceV1Evaluator(model, config)
    test_samples = [{"prompt": "Describe Verace V1", "ground_truth": "Verace"}]
    res = evaluator.evaluate_benchmark("V1-Omni-Bench", test_samples=test_samples)
    assert "accuracy_score" in res
    print(f"Evaluator test passed: {res}")

    print("All Verace V1 end-to-end tests passed.")

if __name__ == "__main__":
    test_end2end_pipeline()
