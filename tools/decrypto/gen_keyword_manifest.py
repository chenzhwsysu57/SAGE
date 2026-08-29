"""Generate keyword-set manifests for Decrypto difficulty tiers.

Tiers:
  all_abstract: 4 abstract words per game
  3_abstract:   3 abstract + 1 concrete
  2_abstract:   2 abstract + 2 concrete

Each tier produces 30 keyword sets with fixed random seed for reproducibility.
Output: analysis/decrypto/keyword_manifest.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent.parent

    # Load classification
    classes = json.loads((root / "analysis" / "decrypto" / "word_classes.json").read_text())
    abstract_words = sorted(w for w, c in classes.items() if c == "abstract")
    concrete_words = sorted(w for w, c in classes.items() if c == "concrete")

    print(f"Abstract pool: {len(abstract_words)}")
    print(f"Concrete pool: {len(concrete_words)}")

    rng = random.Random(42)  # fixed seed for reproducibility

    # Generate manifests
    manifests: dict[str, list[list[str]]] = {}

    for tier_name, n_abstract in [("all_abstract", 4), ("3_abstract", 3), ("2_abstract", 2)]:
        n_concrete = 4 - n_abstract
        sets: list[list[str]] = []
        a_pool = abstract_words.copy()
        c_pool = concrete_words.copy()
        rng.shuffle(a_pool)
        rng.shuffle(c_pool)

        for i in range(30):
            if len(a_pool) < n_abstract or len(c_pool) < n_concrete:
                a_pool = abstract_words.copy()
                c_pool = concrete_words.copy()
                rng.shuffle(a_pool)
                rng.shuffle(c_pool)

            kw_set = a_pool[:n_abstract] + c_pool[:n_concrete]
            a_pool = a_pool[n_abstract:]
            c_pool = c_pool[n_concrete:]
            rng.shuffle(kw_set)
            sets.append(kw_set)

        manifests[tier_name] = sets
        print(f"  {tier_name}: {len(sets)} sets")

    out_path = root / "analysis" / "decrypto" / "keyword_manifest.json"
    out_path.write_text(json.dumps(manifests, indent=2))
    print(f"\nSaved to {out_path}")

    # Show examples
    print("\nExamples:")
    for tier_name in ["all_abstract", "3_abstract", "2_abstract"]:
        print(f"  {tier_name}: {manifests[tier_name][0]}")


if __name__ == "__main__":
    main()
