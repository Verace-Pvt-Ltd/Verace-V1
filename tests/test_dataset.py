"""
Unit tests for TextDataset's on-disk tokenization cache and memmap-backed windowing.
"""
import os
import shutil
import tempfile

import numpy as np
import torch

from verace_v1.dataset.dataset import TextDataset, create_pretrain_dataloader
from verace_v1.tokenizer import VeraceTokenizer


def test_windows_match_whole_file_tokenization():
    """The memmap-backed windows must be byte-identical to tokenizing the whole file
    in one call -- proves the chunked-to-disk tokenization doesn't perturb the token
    stream, only where it's stored."""
    tmp_dir = tempfile.mkdtemp()
    try:
        text = "The quick brown fox jumps over the lazy dog.\n" * 500
        path = os.path.join(tmp_dir, "sample.txt")
        with open(path, "w") as f:
            f.write(text)

        tok = VeraceTokenizer(vocab_size=50000)
        ds = TextDataset(data_path=path, tokenizer=tok, context_length=32, stride=32)
        whole = tok.encode(text)

        assert ds[0]["input_ids"].tolist() == whole[0:32]
        assert ds[1]["input_ids"].tolist() == whole[32:64]
        assert ds[0]["labels"].tolist() == whole[1:33]
        print(f"{len(ds)} windows, first two match whole-file tokenization exactly.")
    finally:
        shutil.rmtree(tmp_dir)


def test_cache_is_created_and_reused():
    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "sample.txt")
        with open(path, "w") as f:
            f.write("Verace V1 architecture test sentence. " * 200)

        tok = VeraceTokenizer(vocab_size=50000)
        ds1 = TextDataset(data_path=path, tokenizer=tok, context_length=16, stride=16)

        cache_dir = os.path.join(tmp_dir, ".verace_token_cache")
        assert os.path.isdir(cache_dir)
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 1 and cache_files[0].endswith(".bin")

        ds2 = TextDataset(data_path=path, tokenizer=tok, context_length=16, stride=16)
        assert os.listdir(cache_dir) == cache_files, "re-loading should not create a new cache file"
        assert ds1[0]["input_ids"].tolist() == ds2[0]["input_ids"].tolist()
        print("Cache created once, reused on second load.")
    finally:
        shutil.rmtree(tmp_dir)


def test_max_tokens_bounds_corpus_size():
    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "sample.txt")
        with open(path, "w") as f:
            f.write("Verace V1 architecture test sentence. " * 5000)

        tok = VeraceTokenizer(vocab_size=50000)
        ds_full = TextDataset(data_path=path, tokenizer=tok, context_length=16, stride=16)
        ds_capped = TextDataset(data_path=path, tokenizer=tok, context_length=16, stride=16, max_tokens=100)

        assert len(ds_capped._token_source) <= 100
        assert len(ds_capped) <= len(ds_full)
        print(f"Capped dataset: {len(ds_capped._token_source)} tokens vs uncapped: {len(ds_full._token_source)}")
    finally:
        shutil.rmtree(tmp_dir)


def test_too_short_corpus_falls_back_to_repeated_window():
    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "tiny.txt")
        with open(path, "w") as f:
            f.write("short text")

        tok = VeraceTokenizer(vocab_size=50000)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = TextDataset(data_path=path, tokenizer=tok, context_length=4096, stride=4096)
            assert any("fewer than" in str(warning.message) for warning in w)
        assert len(ds) == 1
        assert ds[0]["input_ids"].shape[0] == 4096
    finally:
        shutil.rmtree(tmp_dir)


def test_dataloader_end_to_end_with_real_pretrain_step():
    """Full path through create_pretrain_dataloader -> a real training step, confirming
    memmap-backed batches work correctly with train_pretrain_step (not just __getitem__
    in isolation)."""
    from verace_v1.config import VeraceV1Config
    from verace_v1.modules.backbone import VeraceV1Model
    from verace_v1.optimizer.unitary_muon import build_hybrid_optimizer
    from verace_v1.training.pretrain import train_pretrain_step

    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "sample.txt")
        with open(path, "w") as f:
            f.write("Verace V1 architecture test sentence. " * 500)

        config = VeraceV1Config(
            vocab_size=200, hidden_dim=32, num_layers=2, num_heads=2, head_dim=16,
            spectral_dim=8, chams_holographic_dim=8, mcmoe_rank=4, mcmoe_num_components=4,
            max_cognitive_depth=2, min_cognitive_depth=1
        )
        tokenizer = VeraceTokenizer(vocab_size=config.vocab_size)
        dataloader = create_pretrain_dataloader(
            data_path=path, tokenizer=tokenizer, batch_size=2, context_length=16, num_workers=0
        )
        model = VeraceV1Model(config).cuda()
        optimizer = build_hybrid_optimizer(model)

        batch = next(iter(dataloader))
        batch = {k: v.cuda() for k, v in batch.items()}
        ce_loss, mean_depth = train_pretrain_step(model, optimizer, batch, use_amp=False)
        assert ce_loss > 0.0
        print(f"End-to-end memmap dataloader -> training step: loss={ce_loss:.4f}")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_windows_match_whole_file_tokenization()
    test_cache_is_created_and_reused()
    test_max_tokens_bounds_corpus_size()
    test_too_short_corpus_falls_back_to_repeated_window()
    test_dataloader_end_to_end_with_real_pretrain_step()
