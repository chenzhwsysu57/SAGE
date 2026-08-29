"""Probabilistic tools for Decrypto opponent modeling.

KeywordBelief: P(word | hints_seen) via semantic similarity (MiniLM).
DecryptoBeliefState: manages 4 KeywordBelief instances, one per position.
make_embed_fn: factory for embedding functions.
"""

from tools.decrypto.semantic_belief import (
    DecryptoBeliefState,
    KeywordBelief,
    make_embed_fn,
)
from tools.decrypto.interceptor_style import (
    build_interceptor_model,
    build_interceptor_style_prompt,
    InterceptorStyleEstimate,
    InterceptorStyleModel,
    estimate_interceptor_style,
    infer_interceptor_style_llm,
    normalize_interceptor_history,
    reconstruct_round_records,
)

__all__ = [
    "KeywordBelief",
    "DecryptoBeliefState",
    "make_embed_fn",
    "InterceptorStyleEstimate",
    "InterceptorStyleModel",
    "build_interceptor_model",
    "build_interceptor_style_prompt",
    "estimate_interceptor_style",
    "infer_interceptor_style_llm",
    "normalize_interceptor_history",
    "reconstruct_round_records",
]
