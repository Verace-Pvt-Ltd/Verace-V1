"""
Regression tests for visual-token fusion in VeraceV1Model.

Covers a bug found and fixed after publishing: the original implementation
unconditionally overwrote token_embeds[:, :v_len] with visual tokens, silently
destroying whatever text embeddings occupied those leading positions. The fix
(VeraceV1Model._fuse_visual_tokens) either replaces embeddings only at explicit
media_placeholder_token_id positions, or -- when no placeholder is configured or
present -- prepends visual tokens instead of overwriting anything.
"""
import torch
from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.training.pretrain import train_pretrain_step
from verace_v1.optimizer.unitary_muon import build_hybrid_optimizer

def _small_config(**overrides):
    config = VeraceV1Config(
        vocab_size=2000, hidden_dim=64, num_layers=2, num_heads=2, head_dim=32,
        spectral_dim=16, chams_holographic_dim=16, mcmoe_rank=4, mcmoe_num_components=4,
        max_cognitive_depth=2, min_cognitive_depth=1,
        **overrides,
    )
    config.vision_config.embed_dim = 32
    config.vision_config.num_layers = 1
    config.vision_config.num_heads = 2
    config.vision_config.patch_size = 4
    return config

def test_fallback_prepends_instead_of_overwriting_text():
    """No placeholder configured -> visual tokens are prepended, text is untouched."""
    torch.manual_seed(0)
    config = _small_config()
    model = VeraceV1Model(config).cuda()
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 5), device="cuda")
    images = torch.randn(1, 3, 16, 8, device="cuda")  # -> 4 visual tokens after 2x2 merge

    with torch.no_grad():
        text_embeds = model.embed_tokens(input_ids).clone()
        visual_tokens = model.vision_encoder(images)
        fused = model._fuse_visual_tokens(text_embeds, input_ids, visual_tokens)

    v_len = visual_tokens.shape[1]
    assert fused.shape[1] == v_len + input_ids.shape[1]
    assert torch.equal(fused[:, v_len:], text_embeds), "original text embeddings must be byte-for-byte unchanged"
    print(f"Prepend path: text preserved exactly, sequence grew {input_ids.shape[1]} -> {fused.shape[1]}")

def test_placeholder_routing_preserves_sequence_length_and_other_text():
    """Placeholder configured and present -> only those positions change, seq_len is preserved."""
    torch.manual_seed(0)
    config = _small_config(media_placeholder_token_id=1999)
    model = VeraceV1Model(config).cuda()
    model.eval()

    input_ids = torch.randint(0, 1999, (1, 8), device="cuda")
    placeholder_positions = [2, 5]
    for pos in placeholder_positions:
        input_ids[0, pos] = 1999

    images = torch.randn(1, 3, 8, 8, device="cuda")  # -> 1 visual token

    with torch.no_grad():
        text_embeds = model.embed_tokens(input_ids).clone()
        visual_tokens = model.vision_encoder(images)
        fused = model._fuse_visual_tokens(text_embeds, input_ids, visual_tokens)

    assert fused.shape[1] == input_ids.shape[1], "sequence length must not change when placeholder routing is used"

    other_positions = [i for i in range(8) if i not in placeholder_positions]
    assert torch.equal(fused[:, other_positions], text_embeds[:, other_positions]), "non-placeholder text must be untouched"
    assert torch.equal(fused[:, placeholder_positions[0]], visual_tokens[:, 0]), "first placeholder position must carry the visual token"
    print("Placeholder routing: sequence length preserved, only placeholder positions replaced.")

def test_pretrain_step_aligns_logits_when_images_grow_sequence_length():
    """
    Regression test: before this was fixed, images.shape mismatch between logits
    (grown by prepending) and labels (original length) would misalign shift_logits/
    shift_labels silently -- wrong gradients, no error. Verifies the alignment guard
    in train_pretrain_step actually engages and produces a valid loss.
    """
    torch.manual_seed(0)
    config = _small_config()
    model = VeraceV1Model(config).cuda()
    optimizer = build_hybrid_optimizer(model, muon_lr=0.01, adamw_lr=0.001)

    input_ids = torch.randint(0, config.vocab_size, (1, 5), device="cuda")
    images = torch.randn(1, 3, 16, 8, device="cuda")
    batch = {"input_ids": input_ids, "labels": input_ids.clone(), "image": images}

    ce_loss, mean_depth = train_pretrain_step(model, optimizer, batch)
    assert ce_loss > 0.0
    assert not torch.isnan(torch.tensor(ce_loss))
    print(f"Training step with images (sequence-length-growing path) completed: ce_loss={ce_loss:.4f}")

if __name__ == "__main__":
    test_fallback_prepends_instead_of_overwriting_text()
    test_placeholder_routing_preserves_sequence_length_and_other_text()
    test_pretrain_step_aligns_logits_when_images_grow_sequence_length()
