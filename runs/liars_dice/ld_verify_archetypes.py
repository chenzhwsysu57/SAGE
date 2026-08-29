"""Verify archetype behavior: run 5 games per archetype, log action patterns.
Saves prompts by default for manual inspection.

Each game: SimplePrompt(P0) vs ArchetypeAgent(P1).
Outputs per-archetype action table + saved prompts in results/.
"""

import os, sys, time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from games.liars_dice import LiarsDice
from apis.openai_wrapper import OpenAIModel
from methods.simple_prompt import SimplePromptMethod
from modeling.liars_dice.archetypes import ArchetypeAgent, ARCHETYPES, ARCHETYPE_NAMES
from rollout import RolloutRunner

model = OpenAIModel(model="deepseek-v4-flash", base_url="https://api.deepseek.com")

# Output dir
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results/liars_dice/archetype_verify/{TIMESTAMP}")
PROMPTS_DIR = OUTPUT_DIR / "prompts"
METRICS_DIR = OUTPUT_DIR / "metrics"
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def describe_opponent_action(step, opp_id):
    """Extract opponent's action with context from step."""
    if step.player_id != opp_id:
        return None
    action = step.parsed_action
    if action is None:
        return None
    sb = step.state_before
    if sb is None:
        return None
    # Keys are ints (0, 1) in to_dict() output
    opp_dice = sb.get("dice", {}).get(opp_id, [])
    last_bid = sb.get("last_bid")
    dice_counts = sb.get("dice_counts", {})
    my_count = dice_counts.get(opp_id, "?")
    opp_count = dice_counts.get(1 - opp_id, "?")

    # How many of bid face does opponent actually have?
    bid_quantity = action.data.get("quantity") if action.type == "bid" else None
    bid_face = action.data.get("face") if action.type == "bid" else None
    held = sum(1 for d in opp_dice if d == bid_face) if bid_face else None
    overbid = bid_quantity - held if (bid_quantity is not None and held is not None) else None

    return {
        "round": sb.get("round_num", "?"),
        "dice": "".join(str(d) for d in sorted(opp_dice)),
        "my_count": my_count,
        "opp_count": opp_count,
        "last_bid": f"{last_bid[0]}×{last_bid[1]}" if last_bid else "none",
        "action": action.type,
        "bid_q": bid_quantity,
        "bid_f": bid_face,
        "held": held,
        "overbid": overbid,
    }


def save_prompts(archetype, traj, game_idx):
    """Save all prompts & responses for a game."""
    gdir = PROMPTS_DIR / archetype / f"game_{game_idx}"
    gdir.mkdir(parents=True, exist_ok=True)
    for step in traj.steps:
        if not step.rendered_prompt:
            continue
        player = f"P{step.player_id}"
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in player)
        prefix = f"step{step.step:03d}_{safe_name}"
        (gdir / f"{prefix}_prompt.txt").write_text(step.rendered_prompt)
        if step.raw_llm_response:
            (gdir / f"{prefix}_response.txt").write_text(step.raw_llm_response)


def run_games(archetype_name, n_games, start_seed=100):
    """Run N games of SimplePrompt vs ArchetypeAgent. Return opponent actions per game."""
    p0 = SimplePromptMethod(name="SimplePrompt-P0")
    p1 = ArchetypeAgent(archetype_name, name=f"{archetype_name}-P1")

    game_logs = []
    for g in range(n_games):
        seed = start_seed + g
        game = LiarsDice(num_dice=4, num_faces=4, seed=seed)
        runner = RolloutRunner(game, {0: p0, 1: p1}, model, verbose=False)

        t0 = time.time()
        print(f"    Game {g+1} (seed={seed})...", flush=True)
        traj = runner.run()
        elapsed = time.time() - t0
        print(f"      done ({elapsed:.0f}s, {len(traj.steps)} steps)", flush=True)

        # Save prompts & responses
        save_prompts(archetype_name, traj, g + 1)

        # Extract opponent (P1) actions from trajectory
        opp_actions = []
        for step in traj.steps:
            info = describe_opponent_action(step, 1)
            if info:
                opp_actions.append(info)

        outcome = "W" if traj.payout.get(1, 0) > 0 else "L"
        game_logs.append({
            "game": g + 1,
            "seed": seed,
            "outcome": outcome,
            "payout": traj.payout,
            "steps": len(traj.steps),
            "time": round(elapsed, 1),
            "opp_actions": opp_actions,
        })
    return game_logs


def print_table(archetype_name, logs):
    """Print a compact action table."""
    print(f"\n{'='*80}")
    print(f"  {archetype_name} ({len(logs)} games)")
    print(f"{'='*80}")

    total_win = sum(1 for l in logs if l['outcome'] == 'W')
    total_bids = 0
    total_challenges = 0

    for log in logs:
        print(f"\n  Game {log['game']} (seed={log['seed']}, "
              f"{log['outcome']}, payout={log['payout']}, "
              f"{log['steps']} steps, {log['time']}s)")

        # Count actions
        n_bids = sum(1 for a in log['opp_actions'] if a['action'] == 'bid')
        n_chal = sum(1 for a in log['opp_actions'] if a['action'] == 'challenge')
        total_bids += n_bids
        total_challenges += n_chal
        print(f"    Bids: {n_bids}, Challenges: {n_chal}")

        # Action table
        print(f"    {'#':>2} {'Rd':>2} {'Dice':>6} {'LastBid':>9} {'Act':>12} "
              f"{'BidQ':>4} {'BidF':>4} {'Held':>4} {'Over':>5}")
        print(f"    {'-'*55}")
        for i, a in enumerate(log['opp_actions']):
            bq = str(a['bid_q']) if a['bid_q'] is not None else ''
            bf = str(a['bid_f']) if a['bid_f'] is not None else ''
            h = str(a['held']) if a['held'] is not None else ''
            ov = f"+{a['overbid']}" if (a['overbid'] is not None and a['overbid'] > 0) else str(a['overbid'] or '')
            print(f"    {i+1:>2} {a['round']:>2} {a['dice']:>6} {a['last_bid']:>9} "
                  f"{a['action']:>12} {bq:>4} {bf:>4} {h:>4} {ov:>5}")

    # Overbid analysis
    all_overbids = []
    for log in logs:
        for a in log['opp_actions']:
            if a['overbid'] is not None:
                all_overbids.append(a['overbid'])
    overbid_rate = sum(1 for o in all_overbids if o > 0) / len(all_overbids) * 100 if all_overbids else 0
    avg_overbid = sum(o for o in all_overbids if o > 0) / sum(1 for o in all_overbids if o > 0) if any(o > 0 for o in all_overbids) else 0

    print(f"\n  Summary: {total_win}/{len(logs)} wins ({total_win/len(logs)*100:.0f}%), "
          f"{total_bids} bids, {total_challenges} challenges "
          f"({total_bids + total_challenges} total actions)")
    print(f"  Overbids: {sum(1 for o in all_overbids if o > 0)}/{len(all_overbids)} "
          f"({overbid_rate:.0f}%), avg excess={avg_overbid:.1f}")


def run_one_archetype(arch_name, n_games):
    """Wrapper for parallel execution."""
    t0 = time.time()
    logs = run_games(arch_name, n_games)
    dt = time.time() - t0
    return arch_name, logs, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5, help="Games per archetype")
    parser.add_argument("--archetypes", nargs="*", default=ARCHETYPE_NAMES,
                        help="Archetypes to run (default: all)")
    args = parser.parse_args()

    run_names = args.archetypes
    n_games = args.games
    print(f"Verifying {len(run_names)} archetypes × {n_games} games each (parallel)\n", flush=True)
    print(f"Prompts saved to: {PROMPTS_DIR}\n", flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=len(run_names)) as ex:
        futures = {ex.submit(run_one_archetype, name, n_games): name for name in run_names}
        for f in as_completed(futures):
            arch_name, logs, dt = f.result()
            results[arch_name] = (logs, dt)
            print(f"  ✓ {arch_name} done ({dt:.0f}s)", flush=True)

    print(f"\n{'='*80}", flush=True)
    for arch_name in run_names:
        logs, dt = results[arch_name]
        print_table(arch_name, logs)
    print(f"\n{'='*80}", flush=True)
    print(f"  ALL DONE — prompts: {PROMPTS_DIR}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
