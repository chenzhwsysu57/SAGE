"""Semantic Bayesian belief over vocabulary for Decrypto keyword inference.

KeywordBelief: maintains P(word | hints_seen) for one keyword position via
Bayesian update with semantic similarity as the likelihood function.

DecryptoBeliefState: manages 4 KeywordBelief instances, extracts hints from
game observations, and formats prompt injection text for the LLM.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

EmbedFn = Callable[[str], np.ndarray]


def make_embed_fn(model_name: str = "all-MiniLM-L6-v2") -> EmbedFn:
    """Create an embedding function backed by sentence-transformers.

    Args:
        model_name: any SentenceTransformer-compatible model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for semantic belief modeling. "
            "Install with: pip install sentence-transformers"
        )
    model = SentenceTransformer(model_name)
    return lambda word: model.encode(word)


class KeywordBelief:
    """Bayesian P(word | hints) over a fixed vocabulary.

    Update rule for each new hint::

        log P(w | h_new) += cosine_sim(emb(h_new), emb(w)) / tau

    Followed by log-normalization.  Multiple hints from different semantic
    angles intersect to rapidly narrow the candidate set.

    Parameters:
        wordlist: candidate vocabulary (e.g. the 679-word Decrypto list).
        embed_fn: word -> embedding vector.  Used for both vocabulary words
                  and arbitrary hint words.
        tau: temperature.  Lower = sharper updates, faster convergence.
    """

    def __init__(
        self,
        wordlist: list[str],
        embed_fn: EmbedFn,
        tau: float = 0.15,
    ):
        self.words = list(wordlist)
        self.n = len(wordlist)
        self.tau = tau
        self._embed_fn = embed_fn

        # Pre-compute and L2-normalize vocabulary embeddings
        embs = np.array([embed_fn(w) for w in wordlist], dtype=np.float64)
        self._word_embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)

        # Log-uniform prior
        self._log_probs = np.full(self.n, -math.log(self.n), dtype=np.float64)
        self._hints_seen: list[str] = []

    # ── Update ────────────────────────────────────────────────────────────

    def update(self, hint: str) -> None:
        """Bayesian update with one new hint word.

        The hint word itself (if it appears in the vocabulary) is zeroed out
        after the update — Alice would never use a keyword as its own hint.
        """
        hint_emb = np.asarray(self._embed_fn(hint), dtype=np.float64)
        hint_emb = hint_emb / (np.linalg.norm(hint_emb) + 1e-10)
        # Cosine similarities to every vocabulary word
        sims = np.dot(self._word_embs, hint_emb)
        self._log_probs += sims / self.tau

        # Zero out the hint word itself if it exists in the vocabulary
        try:
            idx = self.words.index(hint)
            self._log_probs[idx] = -np.inf
        except ValueError:
            pass  # hint not in vocabulary, no self-match to suppress

        self._normalize()
        self._hints_seen.append(hint)

    def _normalize(self) -> None:
        log_z = _logsumexp(self._log_probs)
        self._log_probs -= log_z

    # ── Query ─────────────────────────────────────────────────────────────

    @property
    def probs(self) -> np.ndarray:
        return np.exp(self._log_probs)

    def top_k(self, k: int = 10) -> list[tuple[str, float]]:
        """Top-k most likely words with probabilities."""
        k = min(k, self.n)
        idx = np.argpartition(-self._log_probs, k - 1)[:k]
        idx = idx[np.argsort(-self._log_probs[idx])]
        return [(self.words[i], float(np.exp(self._log_probs[i]))) for i in idx]

    def entropy(self) -> float:
        """Shannon entropy in nats.  0 = certain, ~6.5 = uniform over 679."""
        p = self.probs
        # Avoid 0 * (-inf) = nan for eliminated words
        mask = p > 0
        return -float(np.sum(p[mask] * self._log_probs[mask]))

    @property
    def n_eliminated(self) -> int:
        """Words with probability below 1e-6 (effectively ruled out)."""
        return int(np.sum(self.probs < 1e-6))

    @property
    def hints_seen(self) -> list[str]:
        return list(self._hints_seen)

    # ── Formatting ────────────────────────────────────────────────────────

    def to_prompt(self, k: int = 5) -> str:
        """Compact prompt-ready string for injection into LLM context."""
        top = self.top_k(k)
        lines = []
        for i, (word, prob) in enumerate(top):
            lines.append(f"    {i + 1}. {word:<16s} {prob:5.1%}")
        if self.n_eliminated > 0:
            lines.append(f"    Eliminated: {self.n_eliminated}/{self.n} words")
        hints = ", ".join(self._hints_seen) if self._hints_seen else "(none)"
        lines.append(f"    Hints: {hints}")
        lines.append(f"    Entropy: {self.entropy():.2f} nats (0 = certain, "
                     f"{math.log(self.n):.1f} = uniform)")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "words": self.words,
            "tau": self.tau,
            "log_probs": self._log_probs.tolist(),
            "hints_seen": self._hints_seen,
        }

    @classmethod
    def from_dict(cls, data: dict, embed_fn: EmbedFn) -> KeywordBelief:
        kb = cls(wordlist=data["words"], embed_fn=embed_fn, tau=data["tau"])
        kb._log_probs = np.array(data["log_probs"], dtype=np.float64)
        kb._hints_seen = list(data["hints_seen"])
        return kb


class DecryptoBeliefState:
    """Manages 4 KeywordBelief instances, one per keyword position (1-4).

    Extracts hints from game observations and formats a ``KEYWORD CANDIDATES``
    section for injection into the LLM decision prompt.

    Usage inside a method::

        state = DecryptoBeliefState(wordlist, embed_fn)
        # in _render_prompt:
        state.update_from_observation(obs)
        prompt_section = state.to_prompt(player_id)
    """

    def __init__(
        self,
        wordlist: list[str],
        embed_fn: EmbedFn,
        tau: float = 0.15,
    ):
        self.wordlist = list(wordlist)
        self.embed_fn = embed_fn
        self.tau = tau
        self.beliefs: dict[int, KeywordBelief] = {
            pos: KeywordBelief(wordlist, embed_fn, tau) for pos in (1, 2, 3, 4)
        }
        # Track how many hints per position have been processed
        self._processed: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

    # ── Update from observation ───────────────────────────────────────────

    def update_from_observation(self, obs) -> int:
        """Scan ``obs.public_info['hint_history']`` for new hints.

        Returns the number of new hints processed.
        """
        hint_history = obs.public_info.get("hint_history", {})
        total_new = 0
        for pos in (1, 2, 3, 4):
            key = str(pos)
            hints = hint_history.get(key, [])
            if not isinstance(hints, list):
                continue
            seen = self._processed.get(pos, 0)
            for hint in hints[seen:]:
                self.beliefs[pos].update(str(hint))
                total_new += 1
            self._processed[pos] = len(hints)
        return total_new

    # ── Prompt injection ──────────────────────────────────────────────────

    def to_prompt(self, player_id: int, k: int = 5) -> str:
        """Formatted belief section for injection into the LLM prompt.

        Labels positions 1-4. For Eve (P2) who doesn't know the keywords,
        the position numbers are the only identifiers available.
        """
        total = sum(len(b.hints_seen) for b in self.beliefs.values())
        if total == 0:
            return ""

        lines = [
            "\n#### KEYWORD CANDIDATES (semantic inference) ####\n",
            "Ranked by semantic similarity of each hint to the full vocabulary:\n",
        ]

        for pos in (1, 2, 3, 4):
            kb = self.beliefs[pos]
            if not kb.hints_seen:
                lines.append(
                    f"  Keyword {pos}: NO HINTS YET — all {len(self.wordlist)} "
                    f"words equally likely.\n"
                )
            else:
                lines.append(f"  Keyword {pos}:")
                lines.append(kb.to_prompt(k))
                lines.append("")

        return "\n".join(lines)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all beliefs and hint trackers."""
        for pos in (1, 2, 3, 4):
            self.beliefs[pos] = KeywordBelief(self.wordlist, self.embed_fn, self.tau)
        self._processed = {1: 0, 2: 0, 3: 0, 4: 0}


# ── Math helpers ──────────────────────────────────────────────────────────

def _logsumexp(x: np.ndarray) -> float:
    c = float(x.max())
    if np.isneginf(c) or np.isnan(c):
        return -math.log(len(x))
    return c + math.log(float(np.sum(np.exp(x - c))))
