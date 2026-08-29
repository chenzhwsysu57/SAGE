"""Classify decrypto words by semantic field width — number of distinct meaning clusters.

A word with 1 cluster (e.g. "giraffe") means all hints converge on the same theme.
A word with 2+ clusters (e.g. "bank" = river/finance) allows hints from unrelated domains.

Output: analysis/decrypto/semantic_width.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apis.openai_wrapper import OpenAIModel


def main():
    word_path = (
        Path(__file__).resolve().parent.parent.parent / "games" / "decrypto_words.txt"
    )
    words = [w.strip() for w in word_path.read_text().splitlines() if w.strip()]

    model = OpenAIModel(model="deepseek-v4-flash", temperature=0.0)

    prompt = (
        "For each word below, count the number of DISTINCT semantic clusters (unrelated meaning domains).\n\n"
        "1 cluster = all meanings cluster around one theme. Examples:\n"
        "  giraffe (tall African animal), helicopter (flying vehicle), football (sport),\n"
        "  happiness (positive emotion), democracy (political system)\n\n"
        "2 clusters = two clearly unrelated meaning domains. Examples:\n"
        "  bank (river edge / financial institution),\n"
        "  mole (animal / spy / skin growth),\n"
        "  charge (accusation / electrical / fee)\n\n"
        "3+ clusters = three or more unrelated domains, or extremely broad concept.\n"
        "  Examples: light (brightness / not heavy / ignite / pale color),\n"
        "  run (move fast / operate / flow / campaign / manage)\n\n"
        "Output exactly ONE word and ONE number per line. Format:\n"
        "word: 1\n"
        "word: 2\n"
        "word: 3\n\n"
        f"Words:\n{chr(10).join(words)}"
    )

    raw = model.generate(prompt)
    classified: dict[str, int] = {}

    for line in raw.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            w = parts[0].strip().lower()
            try:
                n = int(parts[1].strip())
                if n in (1, 2, 3):
                    classified[w] = n
            except ValueError:
                pass

    out_dir = Path(__file__).resolve().parent.parent.parent / "analysis" / "decrypto"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "semantic_width.json"
    out_path.write_text(json.dumps(classified, indent=2))

    c1 = sum(1 for v in classified.values() if v == 1)
    c2 = sum(1 for v in classified.values() if v == 2)
    c3 = sum(1 for v in classified.values() if v == 3)
    print(f"Total: {len(classified)} | 1-cluster: {c1} | 2-cluster: {c2} | 3+-cluster: {c3}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
