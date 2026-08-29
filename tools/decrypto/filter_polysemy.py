"""Filter generated words — keep only those with TRULY distinct meaning domains.

An LLM re-evaluates each word: are the meanings from unrelated domains?
Outputs: analysis/decrypto/polysemous_words_verified.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apis.openai_wrapper import OpenAIModel


def main():
    root = Path(__file__).resolve().parent.parent.parent

    audit_path = root / "analysis" / "decrypto" / "polysemous_words_audit.json"
    audit = json.loads(audit_path.read_text())

    model = OpenAIModel(model="deepseek-v4-flash", temperature=0.0)

    words = sorted(audit.keys())
    batch_size = 50
    verified: dict[str, dict] = {}

    for bi in range(0, len(words), batch_size):
        batch = words[bi:bi + batch_size]
        lines = []
        for w in batch:
            a = audit[w]
            lines.append(f"{w}: {' / '.join(a['glosses'])}")
        word_block = "\n".join(lines)

        prompt = f"""For each word below, determine if it has AT LEAST TWO TRULY DISTINCT meanings from UNRELATED domains.

TRULY DISTINCT means the meanings could give hints that have NO semantic overlap:
  GOOD: "bank" — river edge vs financial institution → UNRELATED. Keep.
  GOOD: "spring" — season vs coiled metal → UNRELATED. Keep.
  GOOD: "mole" — animal vs spy → UNRELATED. Keep.
  BAD: "orange" — fruit vs color → RELATED (the fruit IS that color). Reject.
  BAD: "milk" — cow milk vs almond milk → RELATED (same concept). Reject.
  BAD: "ruby" — gem vs red → RELATED (the gem IS red). Reject.

Output format: one per line.
  word: YES (n domains)
  word: NO

Where n = count of truly distinct meaning domains (≥2).

Words to evaluate:
{word_block}"""

        raw = model.generate(prompt)
        for line in raw.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            w = parts[0].strip().lower()
            verdict = parts[1].strip().upper()
            if w in audit and "YES" in verdict:
                verified[w] = audit[w]

        kept = len([w for w in batch if w in verified])
        print(f"Batch {bi // batch_size + 1}: {kept}/{len(batch)} kept")

    print(f"\nVerified: {len(verified)}/{len(audit)} words have truly distinct meanings")

    # Save
    out_path = root / "analysis" / "decrypto" / "polysemous_words_verified.json"
    out_path.write_text(json.dumps(verified, ensure_ascii=False, indent=2))

    # Update word list
    existing_path = root / "games" / "decrypto_words.txt.bak"
    existing = set(existing_path.read_text().strip().lower().split("\n")) if existing_path.exists() else set()

    # Also include existing wide words
    sw_path = root / "analysis" / "decrypto" / "semantic_width.json"
    if sw_path.exists():
        sw = json.loads(sw_path.read_text())
        existing_wide = {w for w, n in sw.items() if n >= 2}
    else:
        existing_wide = set()

    all_good = sorted(existing_wide | set(verified.keys()))
    word_path = root / "games" / "decrypto_words.txt"
    word_path.write_text("\n".join(all_good))
    print(f"Updated word list: {len(all_good)} words ({len(existing_wide)} existing wide, {len(set(verified.keys()) - existing_wide)} new verified)")


if __name__ == "__main__":
    main()
