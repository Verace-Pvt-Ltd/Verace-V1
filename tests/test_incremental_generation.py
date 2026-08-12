"""
Regression tests for generation-time SSSD/CHAM state caching (layer_states
threaded through VeraceV1Model.forward / AdaptiveCognitiveDepthEngine).

Before this existed, hyper_generate.py recomputed the entire growing sequence
from scratch on every generated token, ignoring the O(1)-per-token recurrent
state SSSD/CHAM are specifically designed to provide. These tests prove the
incremental path is a genuine optimization -- not a different (wrong)
computation -- by checking it against full-sequence recomputation bit-for-bit
(within floating-point tolerance).
"""
import torch

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.serving.hyper_generate import VeraceV1Generator


def _small_config():
    return VeraceV1Config(
        vocab_size=200, hidden_dim=32, num_layers=3, num_heads=2, head_dim=16,
        spectral_dim=8, chams_holographic_dim=8, mcmoe_rank=4, mcmoe_num_components=4,
        max_cognitive_depth=3, min_cognitive_depth=1
    )


def test_incremental_layer_states_match_full_sequence_forward():
    """
    Processing a sequence in two chunks (prefix, then one new token at a time
    with layer_states threaded from the previous call) must produce logits
    for the tail positions numerically matching a single full-sequence
    forward pass over the whole thing at once.
    """
    torch.manual_seed(0)
    config = _small_config()
    model = VeraceV1Model(config).cuda()
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 10), device="cuda")

    with torch.no_grad():
        logits_full, _ = model(input_ids, use_adaptive_depth=True)

        prefix_len = 6
        _, _, layer_states = model(
            input_ids[:, :prefix_len], use_adaptive_depth=True, return_layer_states=True
        )
        chunked_logits = []
        for t in range(prefix_len, input_ids.shape[1]):
            step_logits, _, layer_states = model(
                input_ids[:, t:t + 1], use_adaptive_depth=True,
                return_layer_states=True, layer_states=layer_states
            )
            chunked_logits.append(step_logits)
        chunked_logits = torch.cat(chunked_logits, dim=1)

    torch.testing.assert_close(chunked_logits, logits_full[:, prefix_len:], rtol=1e-3, atol=1e-4)
    print("Incremental (chunked) forward matches full-sequence forward.")


def test_incremental_matches_token_by_token_from_the_start():
    """Same check, but incremental from the very first token (prefix_len=1) --
    the regime hyper_generate.py's main decode loop actually runs in."""
    torch.manual_seed(1)
    config = _small_config()
    model = VeraceV1Model(config).cuda()
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 7), device="cuda")

    with torch.no_grad():
        logits_full, _ = model(input_ids, use_adaptive_depth=True)

        step_logits_0, _, layer_states = model(
            input_ids[:, :1], use_adaptive_depth=True, return_layer_states=True
        )
        chunked_logits = [step_logits_0]
        for t in range(1, input_ids.shape[1]):
            step_logits, _, layer_states = model(
                input_ids[:, t:t + 1], use_adaptive_depth=True,
                return_layer_states=True, layer_states=layer_states
            )
            chunked_logits.append(step_logits)
        chunked_logits = torch.cat(chunked_logits, dim=1)

    torch.testing.assert_close(chunked_logits, logits_full, rtol=1e-3, atol=1e-4)
    print("Token-by-token incremental forward matches full-sequence forward from position 0.")


def test_generate_runs_with_and_without_tree_search():
    """End-to-end smoke test through the actual incremental generate() path,
    both with and without tree search (which now branches off cached
    layer_states instead of recomputing full candidate sequences)."""
    torch.manual_seed(2)
    config = _small_config()
    model = VeraceV1Model(config).cuda()
    model.eval()
    generator = VeraceV1Generator(model, config)

    out_no_tree = generator.generate("hello", max_new_tokens=5, use_tree_search=False)
    out_tree = generator.generate("hello", max_new_tokens=5, use_tree_search=True, num_branches=3)

    assert isinstance(out_no_tree, str)
    assert isinstance(out_tree, str)
    print(f"generate() smoke test passed. no_tree={out_no_tree!r} tree={out_tree!r}")


if __name__ == "__main__":
    test_incremental_layer_states_match_full_sequence_forward()
    test_incremental_matches_token_by_token_from_the_start()
    test_generate_runs_with_and_without_tree_search()
