"""Generate polysemous/abstract word list with Chinese glosses for audit.

Outputs TWO files:
  1. games/decrypto_words.txt — clean word list (game reads this)
  2. analysis/decrypto/polysemous_words_audit.json — word → meanings for human review
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apis.openai_wrapper import OpenAIModel


def main():
    model = OpenAIModel(model="deepseek-v4-flash", temperature=0.7)

    prompt = """Generate 300 English words suitable for a code-guessing game. Each word
needs multiple DISTINCT common meanings from UNRELATED domains.

Output format (one per line):
word: 中文释义1 / 中文释义2 / 中文释义3

Where each 中文释义 is a brief Chinese meaning of one distinct sense.

Examples of GOOD polysemous words:
  spring: 春天 / 弹簧 / 泉水
  match: 比赛 / 火柴 / 匹配
  bank: 河岸 / 银行
  mole: 鼹鼠 / 间谍 / 痣
  charge: 指控 / 充电 / 费用
  light: 光 / 轻的 / 点燃
  run: 跑步 / 经营 / 流动

Examples of GOOD broad abstract words:
  justice: 司法公正 / 报复 / 社会公平
  power: 电力 / 权力 / 数学幂
  freedom: 政治自由 / 个人自由 / 艺术自由

Requirements:
  - 3-8 letters, single word, common vocabulary
  - Meanings MUST be from truly unrelated domains (not variations of the same idea)
  - At least 2 meanings, prefer 3+
  - Each Chinese gloss should be 1-3 characters, precise
  - No proper nouns, brands, or technical terms

Output exactly 300 lines. No numbering, no extra commentary."""

    raw = model.generate(prompt)

    # Parse: "word: gloss1 / gloss2 / gloss3"
    words = []
    audit = {}

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

        words.append(w)
        audit[w] = {"glosses": glosses, "clusters": len(glosses)}

    print(f"Generated: {len(words)} qualifying words")
    c2 = sum(1 for a in audit.values() if a["clusters"] == 2)
    c3 = sum(1 for a in audit.values() if a["clusters"] >= 3)
    print(f"  2-cluster: {c2}, 3+cluster: {c3}")

    # Merge with existing wide words
    root = Path(__file__).resolve().parent.parent.parent
    sw_path = root / "analysis" / "decrypto" / "semantic_width.json"
    if sw_path.exists():
        sw = json.loads(sw_path.read_text())
        existing_wide = {w for w, n in sw.items() if n >= 2}
    else:
        existing_wide = set()

    existing_path = root / "games" / "decrypto_words.txt"
    if existing_path.exists():
        existing_path.rename(existing_path.with_suffix(".txt.bak"))
        print(f"Backed up old word list")

    all_words = sorted(existing_wide | set(words))
    existing_path.write_text("\n".join(all_words))
    print(f"Wrote {len(all_words)} total words ({len(existing_wide)} from existing wide, {len(set(words) - existing_wide)} new)")

    # Save audit
    audit_path = root / "analysis" / "decrypto" / "polysemous_words_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Audit saved to {audit_path}")

    # Show some examples
    print("\nSample generated words:")
    import random
    rng = random.Random(1)
    for w in rng.sample(sorted(audit.keys()), min(15, len(audit))):
        a = audit[w]
        print(f"  {w}: {' / '.join(a['glosses'])} [{a['clusters']} clusters]")


if __name__ == "__main__":
    main()
