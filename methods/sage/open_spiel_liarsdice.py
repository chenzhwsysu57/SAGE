import time
import pickle
import json
import numpy as np
import pyspiel
from tqdm import tqdm

from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import exploitability


GAME_NAME = "liars_dice"
GAME_PARAMS = {
    "players": 2,
    "numdice": 2,
    "dice_sides": 3,
    "bidding_rule": "reset-face",
}

SAVE_PATH = "liarsdice_2x3_mccfr_checkpoint.pkl"
BEST_SAVE_PATH = "liarsdice_2x3_mccfr_best.pkl"
LOG_PATH = "liarsdice_2x3_mccfr_log.jsonl"


def save_checkpoint(path, solver, t, elapsed, exp, game_name, game_params):
    checkpoint = {
        "game_name": game_name,
        "game_params": game_params,
        "iteration": t,
        "elapsed_sec": elapsed,
        "exploitability": exp,
        "infostates": solver._infostates,
        "numpy_random_state": np.random.get_state(),
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


game = pyspiel.load_game(GAME_NAME, GAME_PARAMS)
solver = external_sampling_mccfr.ExternalSamplingSolver(game)

time_budget_sec = 3600
eval_every = 50_000
target = 0.005

start = time.time()
last_pbar_update = 0.0
t = 0
last_exp = None
best_exp = float("inf")

with tqdm(total=time_budget_sec, unit="s", desc="MCCFR training") as pbar:
    while True:
        elapsed = time.time() - start
        if elapsed >= time_budget_sec:
            break

        solver.iteration()
        t += 1

        elapsed = time.time() - start
        delta = elapsed - last_pbar_update

        if delta >= 0.5:
            pbar.update(min(delta, time_budget_sec - pbar.n))
            last_pbar_update = elapsed

            it_per_sec = t / max(elapsed, 1e-9)
            pbar.set_postfix({
                "iter": t,
                "iter/s": f"{it_per_sec:.1f}",
                "exp": "NA" if last_exp is None else f"{last_exp:.6g}",
                "best": "NA" if best_exp == float("inf") else f"{best_exp:.6g}",
            })

        if t % eval_every == 0:
            avg_policy = solver.average_policy()
            last_exp = exploitability.exploitability(game, avg_policy)
            elapsed = time.time() - start
            it_per_sec = t / max(elapsed, 1e-9)

            row = {
                "iter": t,
                "elapsed_sec": elapsed,
                "iter_per_sec": it_per_sec,
                "exploitability": last_exp,
            }

            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")

            # 保存当前 checkpoint
            save_checkpoint(
                SAVE_PATH,
                solver,
                t,
                elapsed,
                last_exp,
                GAME_NAME,
                GAME_PARAMS,
            )

            # 保存 best checkpoint
            if last_exp < best_exp:
                best_exp = last_exp
                save_checkpoint(
                    BEST_SAVE_PATH,
                    solver,
                    t,
                    elapsed,
                    last_exp,
                    GAME_NAME,
                    GAME_PARAMS,
                )

            pbar.set_postfix({
                "iter": t,
                "iter/s": f"{it_per_sec:.1f}",
                "exp": f"{last_exp:.6g}",
                "best": f"{best_exp:.6g}",
            })

            if last_exp < target:
                break

    elapsed = time.time() - start
    if pbar.n < min(elapsed, time_budget_sec):
        pbar.update(min(elapsed, time_budget_sec) - pbar.n)


avg_policy = solver.average_policy()
final_exp = exploitability.exploitability(game, avg_policy)
elapsed = time.time() - start

save_checkpoint(
    SAVE_PATH,
    solver,
    t,
    elapsed,
    final_exp,
    GAME_NAME,
    GAME_PARAMS,
)

if final_exp < best_exp:
    best_exp = final_exp
    save_checkpoint(
        BEST_SAVE_PATH,
        solver,
        t,
        elapsed,
        final_exp,
        GAME_NAME,
        GAME_PARAMS,
    )

print()
print("final_iter:", t)
print("elapsed_sec:", elapsed)
print("final_exploitability:", final_exp)
print("saved_checkpoint:", SAVE_PATH)
print("saved_best_checkpoint:", BEST_SAVE_PATH)
print("log_path:", LOG_PATH)
