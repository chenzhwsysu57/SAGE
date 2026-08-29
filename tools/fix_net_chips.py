#!/usr/bin/env python3
"""
Batch fix net chips statistics across all existing summary.json files.

Detection logic:
  - Leduc: rewards = {winner: pot, loser: 0} (gross pot).
           total_chips = pot_difference. Cannot retroactively compute exact net profit
           without trajectory data (total_bets per player). Will flag and estimate.
  - Liar's Dice ablation (ld_tom_ablation): existing summaries may mix old
           aggregate formulas. Exact fix: recompute aggregates from per-game
           method_chips.
  - Liar's Dice benchmark (ld_cfr_benchmark): already uses sum(method_chips) → correct.
  - Goofspiel (goofspiel_bench): already uses total_method_chips → correct.
"""

import json
import os
import glob

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def fix_ld_ablation_summary(path: str) -> dict | None:
    """Fix Liar's Dice summaries by recomputing totals from per-game rows."""
    with open(path) as f:
        data = json.load(f)

    modified = False
    for c in data.get("conditions", []):
        position_totals = []
        for pos_key in ("p0_results", "p1_results"):
            pos = c.get(pos_key)
            if not pos:
                continue
            rows = pos.get("results") or []
            if not rows or not all("method_chips" in row for row in rows):
                continue
            new_total = sum(float(row["method_chips"]) for row in rows)
            position_totals.append(new_total)
            if pos.get("total_chips") != new_total:
                pos["total_chips"] = new_total
                modified = True
            new_avg = new_total / pos.get("games", 1) if pos.get("games") else 0
            if pos.get("avg_chips") != new_avg:
                pos["avg_chips"] = new_avg
                modified = True

        if position_totals:
            new_total = sum(position_totals)
            if c.get("total_chips") != new_total:
                c["total_chips"] = new_total
                modified = True
            new_avg = new_total / c.get("games", 1) if c.get("games") else 0
            if c.get("avg_chips") != new_avg:
                c["avg_chips"] = new_avg
                modified = True

    if modified:
        return data
    return None


def scan_and_fix():
    """Scan all summary.json files and fix where applicable."""
    summary_files = glob.glob(os.path.join(RESULTS_DIR, "**", "summary.json"), recursive=True)
    summary_files.sort()

    fixed_ld = 0
    skipped_ld_benchmark = 0
    skipped_goofspiel = 0
    skipped_decrypto = 0
    flagged_leduc = 0

    for path in summary_files:
        rel = os.path.relpath(path, RESULTS_DIR)

        # Skip anything outside our target games
        if "/decrypto/" in path or "/werewolf/" in path:
            skipped_decrypto += 1
            continue

        # ── Liar's Dice ablation (ld_tom_ablation) ──
        if "/ld_tom_ablation/" in path:
            result = fix_ld_ablation_summary(path)
            if result:
                with open(path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"FIXED  {rel}  (recomputed total_chips from method_chips)")
                fixed_ld += 1
            else:
                print(f"OK     {rel}  (already correct)")
                skipped_ld_benchmark += 1
            continue

        # ── Liar's Dice benchmark ──
        if "/ld_cfr_benchmark/" in path:
            # Already uses sum(method_chips) → correct net profit
            print(f"OK     {rel}  (ld_cfr_benchmark, already net profit)")
            skipped_ld_benchmark += 1
            continue

        # ── Goofspiel ──
        if "/goofspiel/" in path:
            # goofspiel_bench already uses total_method_chips → correct net
            print(f"OK     {rel}  (goofspiel, already net profit)")
            skipped_goofspiel += 1
            continue

        # ── Leduc ──
        if "/leduc/" in path:
            print(f"FLAG   {rel}  (leduc: total_chips = pot difference, not net profit)")
            flagged_leduc += 1
            continue

        # ── Unknown ──
        print(f"SKIP   {rel}  (unknown experiment type)")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Fixed (LD ablation, recomputed total_chips): {fixed_ld}")
    print(f"  Skipped (LD benchmark, already correct): {skipped_ld_benchmark}")
    print(f"  Skipped (Goofspiel, already correct):     {skipped_goofspiel}")
    print(f"  Skipped (Decrypto/Werewolf):              {skipped_decrypto}")
    print(f"  Flagged (Leduc, cannot fix retroactively): {flagged_leduc}")
    print(f"{'='*60}")
    print()
    print("Leduc runs: total_chips = pot_difference (gross pot won minus lost).")
    print("  To get exact net profit, re-run with fixed leduc_tom_ablation.py")
    print("  which computes payout - total_bets per game from trajectory data.")


if __name__ == "__main__":
    scan_and_fix()
