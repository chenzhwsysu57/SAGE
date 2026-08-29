"""Second-pass audit on rejected words to recover false positives."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apis.openai_wrapper import OpenAIModel


def main():
    root = Path(__file__).resolve().parent.parent.parent

    # Load the final-pass result
    final_path = root / "analysis" / "decrypto" / "polysemous_final.json"
    passed = json.loads(final_path.read_text())

    # Load original clean to find rejected
    clean_path = root / "analysis" / "decrypto" / "polysemous_clean.json"
    original = json.loads(clean_path.read_text())

    rejected = {w: original[w] for w in original if w not in passed}
    print(f"Re-evaluating {len(rejected)} rejected words...")

    model = OpenAIModel(model="deepseek-v4-flash", temperature=0.0)

    words = sorted(rejected.keys())
    batch_size = 50
    recovered: dict[str, dict] = {}

    for bi in range(0, len(words), batch_size):
        batch = words[bi:bi + batch_size]
        lines = []
        for w in batch:
            a = original[w]
            lines.append(f"{w}: {' / '.join(a['glosses'])}")

        prompt = """Re-evaluate each word. Be PRECISE: only REJECT if ALL meanings share
the same semantic core. KEEP if any two meanings come from UNRELATED domains.

KEEP examples:
  fire: flame / shooting / dismiss from job → KEEP (flame vs dismiss = unrelated)
  fall: autumn / drop from height → KEEP (season vs physical action = unrelated)
  flat: level surface / apartment → KEEP (geometry vs housing = unrelated)
  bridge: structure over water / card game → KEEP (engineering vs game = unrelated)
  book: bound pages / make a reservation → KEEP (object vs action = unrelated)
  current: flow of electricity / present time → KEEP (physics vs time = unrelated)
  bear: large animal / endure hardship → KEEP (animal vs verb = unrelated)
  field: open land / area of study → KEEP (physical vs academic = unrelated)
  face: front of head / confront → KEEP (body part vs action = unrelated)
  drive: operate vehicle / motivation → KEEP (action vs abstract = unrelated)

REJECT examples (meanings share same core):
  cabin: small house / airplane cabin / ship cabin → REJECT (all "small room")
  elbow: body part / pipe bend → REJECT (both "bend/angle")
  finger: body part / point out → REJECT (point with finger → same source)
  entrance: doorway / captivate → REJECT (very different, but "entrance=doorway" and "entrance=captivate" are homographs not different meanings of same word for most speakers)
  express: say / fast train → REJECT (fast train is derived from "express=fast", related)

Output: word: KEEP or word: REJECT
""" + "\n".join(lines)

        raw = model.generate(prompt)
        for line in raw.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            w = parts[0].strip().lower()
            verdict = parts[1].strip().upper()
            if w not in rejected:
                continue
            if verdict.startswith("KEEP") or "KEEP" in verdict:
                recovered[w] = rejected[w]

        print(f"  Batch {bi // batch_size + 1}: {sum(1 for w in batch if w in recovered)}/{len(batch)} recovered")

    # Merge: passed + recovered
    all_passed = dict(passed)
    all_passed.update(recovered)
    print(f"\nFinal: {len(passed)} originally kept + {len(recovered)} recovered = {len(all_passed)} total")

    # Save
    out_path = root / "analysis" / "decrypto" / "polysemous_final.json"
    out_path.write_text(json.dumps(all_passed, ensure_ascii=False, indent=2))

    word_path = root / "games" / "decrypto_words.txt"
    word_path.write_text("\n".join(sorted(all_passed.keys())))
    print(f"Updated word list: {len(all_passed)} words")


if __name__ == "__main__":
    main()
