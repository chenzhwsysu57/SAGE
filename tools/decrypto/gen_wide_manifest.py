"""Generate keyword-set manifests filtered by semantic field width.

Tiers:
  all_wide: 4 keywords with ≥2 semantic clusters
  wide_3:   3 wide + 1 narrow
  wide_2:   2 wide + 2 narrow

Uses semantic_width.json classification.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent.parent

    sw = json.loads((root / "analysis" / "decrypto" / "semantic_width.json").read_text())
    wide_words = sorted(w for w, n in sw.items() if n >= 2)
    narrow_words = sorted(w for w, n in sw.items() if n == 1)

    print(f"Wide (≥2 clusters): {len(wide_words)}")
    print(f"Narrow (1 cluster): {len(narrow_words)}")

    rng = random.Random(42)

    manifests: dict[str, list[list[str]]] = {}

    for tier_name, n_wide in [("all_wide", 4), ("wide_3", 3), ("wide_2", 2)]:
        n_narrow = 4 - n_wide
        sets: list[list[str]] = []
        w_pool = wide_words.copy()
        n_pool = narrow_words.copy()
        rng.shuffle(w_pool)
        rng.shuffle(n_pool)

        for i in range(30):
            if len(w_pool) < n_wide or (n_narrow > 0 and len(n_pool) < n_narrow):
                w_pool = wide_words.copy()
                n_pool = narrow_words.copy()
                rng.shuffle(w_pool)
                rng.shuffle(n_pool)

            kw_set = w_pool[:n_wide] + n_pool[:n_narrow]
            w_pool = w_pool[n_wide:]
            n_pool = n_pool[n_narrow:]
            rng.shuffle(kw_set)
            sets.append(kw_set)

        manifests[tier_name] = sets

    # Merge with existing abstract tiers to create comprehensive manifest
    existing = {}
    existing_path = root / "analysis" / "decrypto" / "keyword_manifest.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())

    existing.update(manifests)
    existing_path.write_text(json.dumps(existing, indent=2))
    print(f"\nUpdated {existing_path}")
    for tier_name in ["all_wide", "wide_3", "wide_2"]:
        print(f"  {tier_name}: {len(manifests[tier_name])} sets")
        print(f"    example: {manifests[tier_name][0]}")


if __name__ == "__main__":
    main()
