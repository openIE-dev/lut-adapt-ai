"""Pytest Invariants for Real Polyglot Source-Code Scaling (`PagedBenesLUT6CodeDrafter` + `Mamba3CodeLM`)."""

from __future__ import annotations

import numpy as np
import torch

from clai.contracts.joule_meter import select_torch_device
from clai.frontier.benes_lut_fabric import eval_multicast_bcb512
from clai.frontier.code_corpus_speculative import (
    Mamba3CodeLM,
    PagedBenesLUT6CodeDrafter,
    PLDInPromptLookup,
    RESTCorpusRetrievalDrafter,
    encode_code_bitbus_512,
    load_repo_code_splits,
    speculative_decode_real_code,
)


def test_mamba3_code_lm_chunk_parallel_matches_stateful_step() -> None:
    """Proves exact equivalence between Q=32 GPU matrix scan and T=1..5 stateful verification."""
    dev = torch.device("cpu")
    torch.manual_seed(9)
    lm = Mamba3CodeLM(d_model=128, n_layers=3, n_heads=4).to(dev).eval()
    seq = torch.randint(0, 256, (2, 64), device=dev)

    with torch.no_grad():
        lg_par, _ = lm.forward_train_parallel(seq, chunk_q=32)
        st = lm.init_states(2, dev)
        lg_a, _, st_a = lm.forward_verify_steps(seq[:, :29], st)
        lg_b, _, _ = lm.forward_verify_steps(seq[:, 29:], st_a[-1])
        lg_split = torch.cat([lg_a, lg_b], dim=1)

    assert (lg_par - lg_split).abs().max().item() < 1e-4


def test_paged_benes_lut6_beats_pld_and_rest_on_real_unseen_repo_files() -> None:
    """Proves file-disjoint lossless speculative speedup & Beneš-512 routing parity on real code."""
    dev = select_torch_device()
    torch.manual_seed(42)
    train_files, test_files = load_repo_code_splits()
    assert len(train_files) >= 30 and len(test_files) >= 8
    train_paths = {f.rel_path for f in train_files}
    test_paths = {f.rel_path for f in test_files}
    assert train_paths.isdisjoint(test_paths), "Train and held-out test source files must be disjoint!"

    # Train compact Mamba3CodeLM for 90 steps on real train_files
    rng = np.random.default_rng(42)
    wins = [
        sf.raw_bytes[s : s + 129]
        for sf in train_files
        for s in range(0, len(sf.raw_bytes) - 130, 96)
    ]
    win_arr = np.stack(wins, axis=0)
    lm = Mamba3CodeLM(d_model=192, n_layers=3, n_heads=6).to(dev)
    opt = torch.optim.AdamW(lm.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(90):
        idx = rng.integers(0, len(win_arr), size=(64,))
        bt = torch.from_numpy(win_arr[idx]).to(dev, dtype=torch.long)
        opt.zero_grad()
        lg, _ = lm.forward_train_parallel(bt[:, :128], chunk_q=32)
        torch.nn.functional.cross_entropy(lg.reshape(-1, 256), bt[:, 1:129].reshape(-1)).backward()
        opt.step()
    lm.eval()

    pld = PLDInPromptLookup(draft_k=4)
    rest = RESTCorpusRetrievalDrafter(draft_k=4)
    rest.build_datastore(train_files)
    lut_drafter = PagedBenesLUT6CodeDrafter(draft_k=4)
    lut_drafter.compile_from_corpus(train_files, max_pages4=1000)

    # Verify hottest code LUT6 page compiles through 512x512 Beneš-Copy-Beneš network cleanly
    desired_480, bcb_cfg = lut_drafter.export_hottest_page_to_benes512()
    bus_sample = encode_code_bitbus_512(bytes(test_files[0].raw_bytes[:96].tolist()), test_files[0].lang_id)[None, :]
    assert np.array_equal(eval_multicast_bcb512(bus_sample, bcb_cfg)[:, :480], bus_sample[:, desired_480])

    # Evaluate on 24 unseen 128-byte prefixes from held-out test_files
    prompts = [(test_files[i % len(test_files)].raw_bytes[32 : 160], test_files[i % len(test_files)].lang_id) for i in range(24)]

    ref_outs, _, _ = speculative_decode_real_code(lm, prompts, 48, lambda _c, _l, _h: [], dev)
    pld_outs, pld_bpp, _ = speculative_decode_real_code(lm, prompts, 48, lambda c, _l, _h: pld.draft(c), dev)
    rest_outs, rest_bpp, _ = speculative_decode_real_code(lm, prompts, 48, lambda c, _l, _h: rest.draft(c), dev)
    lut_outs, lut_bpp, _ = speculative_decode_real_code(lm, prompts, 48, lambda c, l, _h: lut_drafter.draft(c, l), dev)

    for a, b, c_o in zip(pld_outs, rest_outs, lut_outs):
        assert len(a) == 48 and len(b) == 48 and len(c_o) == 48
    for o_lut, o_ref in zip(lut_outs, ref_outs):
        assert np.array_equal(o_lut, o_ref), "LUT6 code drafter violated lossless equivalence!"

    assert lut_bpp > 1.5, f"Expected real-code LUT6 speedup > 1.5x, got {lut_bpp:.2f}"
    assert lut_bpp >= max(pld_bpp, rest_bpp), f"LUT6 ({lut_bpp:.2f}) did not beat PLD ({pld_bpp:.2f}) / REST ({rest_bpp:.2f})!"
