"""Classify decrypto word list into abstract vs concrete via LLM.

Saves word_classes.json to analysis/decrypto/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apis.openai_wrapper import OpenAIModel


def main():
    word_path = Path(__file__).resolve().parent.parent.parent / "games" / "decrypto_words.txt"
    words = [w.strip() for w in word_path.read_text().splitlines() if w.strip()]

    model = OpenAIModel(model="deepseek-v4-flash", temperature=0.0)

    chunks = [words[i:i + 200] for i in range(0, len(words), 200)]
    all_classified: dict[str, str] = {}

    for ci, chunk in enumerate(chunks):
        prompt = (
            'Classify each word as "abstract" or "concrete".\n'
            "Abstract = concept, quality, idea, state, process (love, freedom, analysis).\n"
            "Concrete = physical object, person, animal, place, tangible thing (table, dog, river).\n\n"
            "Output EXACTLY one word and label per line. Format:\n"
            "word: abstract\n"
            "word: concrete\n\n"
            f"Words:\n{chr(10).join(chunk)}"
        )

        raw = model.generate(prompt)
        for line in raw.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                w = parts[0].strip().lower()
                label = parts[1].strip().lower()
                if label in ("abstract", "concrete"):
                    all_classified[w] = label

        print(f"Chunk {ci + 1}/{len(chunks)}: {len([w for w in chunk if w.lower() in all_classified])}/{len(chunk)} classified")

    out_dir = Path(__file__).resolve().parent.parent.parent / "analysis" / "decrypto"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "word_classes.json"
    out_path.write_text(json.dumps(all_classified, indent=2))

    a = sum(1 for v in all_classified.values() if v == "abstract")
    c = sum(1 for v in all_classified.values() if v == "concrete")
    print(f"\nTotal: {len(all_classified)} | Abstract: {a} | Concrete: {c}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
