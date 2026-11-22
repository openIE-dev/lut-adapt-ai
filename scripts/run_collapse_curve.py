"""64-Episode Streaming Collapse Curve Benchmark Runner + KV260 LUT6 Verilog Export."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import yaml
from tqdm import trange

from clai.contracts.joule_meter import meter_joules, select_torch_device
from clai.contracts.rce_gate import ModularConstraintEnv
from clai.rung0.verilog_export import export_xilinx_lut6_verilog
from clai.runtime.mixture_of_limits import MixtureOfLimitsRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLAI 4-Rung Collapse Curve Benchmark")
    parser.add_argument("--config", default="configs/collapse_curve.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.episodes is not None:
        cfg["experiment"]["episodes"] = args.episodes

    mb_size = cfg["experiment"].get("microbatch_size", 64)
    total_eps = (cfg["experiment"]["episodes"] // mb_size) * mb_size
    n_steps = total_eps // mb_size
    mutate_every_eps = (cfg["experiment"]["mutation_interval"] // mb_size) * mb_size
    novel_class_ep = (cfg["experiment"].get("novel_class_episode", 0) // mb_size) * mb_size
    compile_every_eps = max(mb_size, (cfg["experiment"]["compile_every"] // mb_size) * mb_size)

    device = select_torch_device() if args.device is None else __import__("torch").device(args.device)
    env = ModularConstraintEnv(seed=cfg["experiment"]["seed"])
    runtime = MixtureOfLimitsRuntime(cfg, env, device)

    r2_params = sum(p.numel() for p in runtime.rung2.parameters()) / 1e6
    r3_params = sum(p.numel() for p in runtime.ebrm.parameters()) / 1e6
    print(
        f"[CLAI] 64-Lane Collapse Curve on {device} ({total_eps} episodes) | "
        f"Rung 2 Mamba-3 MIMO: {r2_params:.2f}M | Rung 3 JEPA-EBRM: {r3_params:.2f}M"
    )

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "collapse_curve_metrics.csv"

    win_j = 0.0
    win_ver = win_r0 = win_r2 = win_soinn = win_r3 = win_eps = 0
    ever_compiled_lut_state = None

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "episode",
                "j_per_vc",
                "r0_satiation_pct",
                "r2_settle_pct",
                "r25_soinn_pct",
                "r3_settle_pct",
                "verified_pass_rate",
                "lut_holdout_agreement",
                "soinn_active_nodes",
            ]
        )

        for step_idx in trange(1, n_steps + 1, desc="64-Ep Micro-Batches"):
            ep_prev = (step_idx - 1) * mb_size
            ep_done = step_idx * mb_size

            if ep_prev > 0 and ep_prev % mutate_every_eps == 0:
                env.mutate_regime()
                print(
                    f"\n[MUTATION] Episode {ep_prev}: Rule schema shifted -> Regime {env.regime_id}"
                )

            if novel_class_ep > 0 and ep_prev == novel_class_ep:
                new_cls = env.spawn_novel_class()
                print(
                    f"\n[OPEN-WORLD] Episode {ep_prev}: Spawned unseen Novel Class ID {new_cls} "
                    f"(total classes={env.n_classes})"
                )

            episodes, _ = env.sample_microbatch(batch_size=mb_size)
            rec = runtime.step_microbatch(episodes)

            win_j += rec.total_joules
            win_ver += rec.verified_count
            win_r0 += rec.rung0_satiated
            win_r2 += rec.rung2_settled
            win_soinn += rec.soinn_settled
            win_r3 += rec.rung3_settled
            win_eps += mb_size

            if ep_done % compile_every_eps == 0:
                with meter_joules(mode="compile", device=device) as m_comp:
                    comp = runtime.crystallizer.compile_downward(
                        batch_size=cfg["compiler"]["batch_size"]
                    )
                _ = m_comp.total_joules
                if runtime.lut_net.layers[0].is_compiled:
                    ever_compiled_lut_state = {
                        k: v.detach().clone() for k, v in runtime.lut_net.state_dict().items()
                    }

                j_per_vc = win_j / max(1, win_ver)
                r0_pct = 100.0 * win_r0 / max(1, win_eps)
                r2_pct = 100.0 * win_r2 / max(1, win_eps)
                r25_pct = 100.0 * win_soinn / max(1, win_eps)
                r3_pct = 100.0 * win_r3 / max(1, win_eps)
                v_rate = win_ver / max(1, win_eps)

                writer.writerow(
                    [
                        ep_done,
                        f"{j_per_vc:.6f}",
                        f"{r0_pct:.2f}",
                        f"{r2_pct:.2f}",
                        f"{r25_pct:.2f}",
                        f"{r3_pct:.2f}",
                        f"{v_rate:.4f}",
                        f"{comp['agreement']:.4f}",
                        runtime.soinn.num_active,
                    ]
                )
                csv_file.flush()

                print(
                    f"\n[Ep {ep_done:5d}] J/VC: {j_per_vc:8.5f} J | "
                    f"R0(LUT6): {r0_pct:5.1f}% | R2(Mamba3): {r2_pct:5.1f}% | "
                    f"R2.5(SOINN): {r25_pct:5.1f}% ({runtime.soinn.num_active} nodes) | "
                    f"R3(EBRM): {r3_pct:5.1f}% | LUT Holdout: {comp['agreement'] * 100:5.1f}%"
                )
                win_j = 0.0
                win_ver = win_r0 = win_r2 = win_soinn = win_r3 = win_eps = 0

    if not runtime.lut_net.layers[0].is_compiled and ever_compiled_lut_state is not None:
        runtime.lut_net.load_state_dict(ever_compiled_lut_state)
        runtime.lut_net.layers[0].is_compiled = True

    if runtime.lut_net.layers[0].is_compiled:
        verilog_path = out_dir / "clai_rung0_kv260.v"
        verilog_path.write_text(export_xilinx_lut6_verilog(runtime.lut_net), encoding="utf-8")
        print(f"[CLAI] Synthesizable Xilinx LUT6 Verilog written to {verilog_path}")


if __name__ == "__main__":
    main()
