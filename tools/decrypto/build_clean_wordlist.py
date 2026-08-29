"""Build a clean word list of genuinely polysemous words via iterative LLM generation + verification.

Keeps the existing verified words, generates more, and runs verification rounds
until the list stabilizes with 300+ truly polysemous words.

Outputs:
  analysis/decrypto/polysemous_clean.json — verified words with glosses
  games/decrypto_words.txt — clean word list for the game
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apis.openai_wrapper import OpenAIModel


VERIFY_PROMPT = """For each word below, determine if it has AT LEAST TWO TRULY DISTINCT meanings from UNRELATED domains.

TRULY DISTINCT (keep):
  "bank" = river edge / financial institution → unrelated. YES
  "spring" = season / coiled metal / water source → unrelated. YES
  "mole" = animal / spy / skin growth → unrelated. YES
  "light" = brightness / not heavy / ignite → unrelated. YES

NOT distinct (reject):
  "orange" = fruit / color → related (the fruit IS that color). NO
  "lobster" = animal / lobster meat → same concept. NO
  "donkey" = animal / stubborn person → metaphor derived from animal. NO
  "ivory" = material / ivory color → color from material. NO
  "crab" = animal / crab meat → same concept. NO
  "silver" = metal / silver color → color from metal. NO
  "pig" = animal / greedy person → metaphor from animal. NO
  "dog" = animal / dog breed → same concept. NO

Output one per line:
  word: YES (N unrelated domains)
  word: NO

Words to evaluate:
{words}"""


def verify_batch(model: OpenAIModel, words_batch: list[str], existing: dict) -> dict:
    """Verify a batch of words. Returns dict of {word: keep?}."""
    to_check = [(w, existing[w]) for w in words_batch if w in existing]
    if not to_check:
        return {}

    lines = []
    for w, info in to_check:
        lines.append(f"{w}: {' / '.join(info['glosses'])}")

    prompt = VERIFY_PROMPT.format(words="\n".join(lines))
    raw = model.generate(prompt)

    results = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        parts = line.split(":", 1)
        w = parts[0].strip().lower()
        verdict = parts[1].strip().upper()
        if "YES" in verdict:
            results[w] = True
        elif "NO" in verdict:
            results[w] = False
    return results


def main():
    root = Path(__file__).resolve().parent.parent.parent
    model = OpenAIModel(model="deepseek-v4-flash", temperature=0.7)
    verify_model = OpenAIModel(model="deepseek-v4-flash", temperature=0.0)

    # Load existing verified word glosses
    audit_path = root / "analysis" / "decrypto" / "polysemous_words_verified.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    clean: dict[str, dict] = {}  # final clean list

    # ── Round 1: Verify existing words ──
    print("=== Round 1: Verifying existing words ===")
    batch_size = 60
    words = sorted(audit.keys())
    kept, rejected = 0, 0
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        results = verify_batch(verify_model, batch, audit)
        for w in batch:
            if w in results and results[w]:
                clean[w] = audit[w]
                kept += 1
            elif w in results:
                rejected += 1
        print(f"  Batch {i // batch_size + 1}: {len(batch)} -> {sum(1 for w in batch if w in results and results[w])} kept, {sum(1 for w in batch if w in results and not results[w])} rejected")
    print(f"  Round 1 result: {kept} kept, {rejected} rejected")

    # ── Rounds 2-N: Generate more + verify, repeat until 300+ ──
    round_num = 2
    while len(clean) < 300:
        print(f"\n=== Round {round_num}: Generating more words ===")
        existing_words = set(clean.keys())

        gen_prompt = f"""Generate 200 English words. Each must have 2-4 TRULY DISTINCT common meanings from UNRELATED domains.

TRUE polysemy (keep generating these):
  bank = river edge / financial institution
  spring = season / coiled metal / water source
  mole = animal / spy / skin growth
  light = brightness / not heavy / ignite
  charge = accusation / electricity / fee
  match = sports game / fire starter / pair
  run = move fast / manage / flow
  scale = fish scales / weighing device / proportion
  court = legal place / sports field / royal residence
  crane = bird / construction machine

FAKE polysemy (DO NOT generate these):
  - Animal → meat: "lobster" = animal / lobster meat (same thing)
  - Animal → metaphor: "donkey" = animal / stubborn (metaphor, not distinct)
  - Material → color: "ivory" = material / ivory color (color from material)
  - Synonyms split: don't use Chinese synonyms to inflate meaning count
  - Same root: meanings that share the same etymology and semantic core

Already have these words, DO NOT repeat:
{', '.join(sorted(existing_words))}

Output format:
  word: Chinese_gloss1 / Chinese_gloss2 / Chinese_gloss3
  One per line. No numbering. Only words with 2+ TRULY UNRELATED domains."""

        raw = model.generate(gen_prompt)

        new_words: dict[str, dict] = {}
        new_only = 0
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":", 1)
            w = parts[0].strip().lower()
            glosses_raw = parts[1].strip()
            glosses = [g.strip() for g in glosses_raw.split("/") if g.strip()]

            if not w.isalpha() or len(w) < 3 or len(w) > 12:
                continue
            if len(glosses) < 2:
                continue
            if w in existing_words:
                continue

            new_words[w] = {"glosses": glosses, "clusters": len(glosses)}
            new_only += 1

        print(f"  Generated {new_only} new candidates")

        # Verify new words
        print(f"  Verifying...")
        new_kept = 0
        new_list = sorted(new_words.keys())
        for i in range(0, len(new_list), batch_size):
            batch = new_list[i:i + batch_size]
            results = verify_batch(verify_model, batch, new_words)
            for w in batch:
                if w in results and results[w]:
                    clean[w] = new_words[w]
                    new_kept += 1

        print(f"  Round {round_num}: +{new_kept} verified, total={len(clean)}")
        round_num += 1

        if round_num > 8:
            print("  Max rounds reached, stopping.")
            break

    # ── Save ──
    clean_path = root / "analysis" / "decrypto" / "polysemous_clean.json"
    clean_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(clean)} clean words to {clean_path}")

    # Update game word list
    word_path = root / "games" / "decrypto_words.txt"
    word_path.write_text("\n".join(sorted(clean.keys())))
    print(f"Updated game word list: {len(clean)} words")

    # Stats
    c2 = sum(1 for a in clean.values() if a['clusters'] == 2)
    c3 = sum(1 for a in clean.values() if a['clusters'] >= 3)
    print(f"  2-cluster: {c2}, 3+cluster: {c3}")


if __name__ == "__main__":
    main()
