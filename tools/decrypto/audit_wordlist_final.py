"""Final audit: keep only words with TRULY distinct, unrelated meaning domains.

For each word, verify: could the meanings produce hints that DON'T converge on a
single theme? If Eve can easily connect all meanings → reject.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apis.openai_wrapper import OpenAIModel


def main():
    root = Path(__file__).resolve().parent.parent.parent

    clean_path = root / "analysis" / "decrypto" / "polysemous_clean.json"
    existing = json.loads(clean_path.read_text())

    model = OpenAIModel(model="deepseek-v4-flash", temperature=0.0)

    # Audit in batches
    words = sorted(existing.keys())
    batch_size = 60
    passed: dict[str, dict] = {}
    rejected: list[str] = []

    for bi in range(0, len(words), batch_size):
        batch = words[bi:bi + batch_size]
        lines = []
        for w in batch:
            a = existing[w]
            glosses_str = " / ".join(a["glosses"])
            lines.append(f"{w}: {glosses_str}")

        prompt = """For each word, determine if its meanings are TRULY DISTINCT — from
completely unrelated domains — such that hints from different meanings
would NOT converge on a single theme for an observer.

KEEP — meanings from UNRELATED domains:
  spring: season / coiled metal / water source (keep — unrelated)
  mole: animal / spy / skin growth (keep — unrelated)
  bank: river edge / financial institution (keep — unrelated)
  light: brightness / not heavy / ignite (keep — unrelated)
  match: sports game / fire starter / pair (keep — unrelated)
  charge: accusation / electricity / fee (keep — unrelated)

REJECT — meanings are RELATED or same domain:
  option: choice / stock option / menu option (reject — all "selection" domain)
  place: location / spot / position (reject — all location synonyms)
  site: location / website / construction site (borderline — website is distinct, but "location/site" overlap. REJECT)
  horse: animal / cavalry / horsepower (reject — all derived from the animal)
  orange: fruit / color / orange tree (reject — color from fruit, tree produces fruit)
  silver: metal / silver medal / silver color (reject — all from the metal)
  donkey: animal / stubborn / fool (reject — metaphors from animal trait)
  lobster: animal / lobster meat / red-faced (reject — meat from animal)
  ivory: material / ivory color / piano key (reject — all from elephant tusk)
  dog: animal / canine / track (reject — canine=dog synonym, track=derived)

Output format, one per line:
  word: KEEP
  word: REJECT (reason)

Words to evaluate:
""" + "\n".join(lines)

        raw = model.generate(prompt)
        for line in raw.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            w = parts[0].strip().lower()
            verdict = parts[1].strip().upper()

            if w in existing:
                if verdict.startswith("KEEP") or "KEEP" in verdict:
                    passed[w] = existing[w]
                else:
                    rejected.append(w)

        kept_in_batch = sum(1 for w in batch if w in passed)
        print(f"Batch {bi // batch_size + 1}: {kept_in_batch}/{len(batch)} kept, "
              f"{sum(1 for w in batch if w in rejected)} rejected")

    print(f"\nFinal: {len(passed)} passed, {len(rejected)} rejected")

    # Show rejected for review
    print(f"\nRejected ({len(rejected)}):")
    for w in sorted(rejected)[:40]:
        print(f"  {w}: {existing[w]['glosses'][0]} / {existing[w]['glosses'][1]} / {existing[w].get('glosses', ['?'])[-1]}")

    # Save
    out_path = root / "analysis" / "decrypto" / "polysemous_final.json"
    out_path.write_text(json.dumps(passed, ensure_ascii=False, indent=2))

    # Update game word list
    word_path = root / "games" / "decrypto_words.txt"
    word_path.write_text("\n".join(sorted(passed.keys())))
    print(f"\nUpdated word list: {len(passed)} words")


if __name__ == "__main__":
    main()
