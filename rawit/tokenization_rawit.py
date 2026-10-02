# Copyright 2025 Wisu Suntoyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RawitTokenizer — SEA-LION Gemma3 tokenizer wrapper with neutral option markers.

Registers [OPT_0] .. [OPT_{MAX_OPTIONS-1}] and [RUBRIC] as additional special tokens.
These abstract markers are injected before each option description in the input sequence,
ensuring the decision head reads option representations from invariant token positions
rather than from the lexical content of option names (polarity-bias fix).

Sequence format produced by encode_decision():
  [BOS] context [SEP] rubric [OPT_0] desc_0 [OPT_1] desc_1 ... [EOS]
"""

import threading
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Maximum number of option marker tokens registered
MAX_OPTIONS = 32
RUBRIC_TOKEN = "[RUBRIC]"
OPTION_TOKEN_TEMPLATE = "[OPT_{i}]"

# Tokenizer access lock: fast tokenizers are not thread-safe when truncation/padding
# settings are mutated (they share a Rust backing object). Serialise all encode calls.
_TOKENIZE_LOCK = threading.RLock()


def _opt_token(i: int) -> str:
    return OPTION_TOKEN_TEMPLATE.format(i=i)


def option_tokens(n: int = MAX_OPTIONS) -> List[str]:
    """Return the list of [OPT_0] .. [OPT_{n-1}] token strings."""
    return [_opt_token(i) for i in range(n)]


def load_tokenizer(
    backbone_name: str = "aisingapore/SEA-LION-ModernBERT-300M",
    revision: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    """Load the SEA-LION tokenizer and register Rawit special tokens.

    All option markers and [RUBRIC] are added as additional_special_tokens so
    the tokenizer assigns them dedicated, non-colliding token ids and never
    splits them into subwords.
    """
    kw = {}
    if revision:
        kw["revision"] = revision
    tok = AutoTokenizer.from_pretrained(backbone_name, **kw)

    new_tokens = option_tokens(MAX_OPTIONS) + [RUBRIC_TOKEN]
    # add_special_tokens returns the count of tokens actually added; already-registered
    # tokens are skipped silently, so re-loading is safe.
    tok.add_special_tokens({"additional_special_tokens": new_tokens})
    return tok


def encode_with_lock(tok, text: str, **kwargs):
    """Tokenize text while holding the shared tokenizer lock."""
    with _TOKENIZE_LOCK:
        return tok(text, **kwargs)


def encode_decision(
    tok,
    context: str,
    rubric: str,
    options: List[str],
    max_len: int = 1024,
    head_max_len: int = 256,
    truncate_context_left: bool = False,
) -> Tuple[List[int], List[int]]:
    """Encode a decision request into token ids and option marker positions.

    Returns:
        input_ids: Full token id sequence (length <= max_len).
        marker_positions: Token index of each [OPT_k] marker in input_ids.
            Length equals len(options). Used by RawitDecisionHead to gather
            per-option hidden states.

    The context is truncated to whatever room remains after the head (rubric +
    option markers + descriptions). Truncation takes from the left for
    conversation turns (truncate_context_left=True) or from the right (default).
    """
    if len(options) > MAX_OPTIONS:
        raise ValueError(
            f"Rawit supports at most {MAX_OPTIONS} options; got {len(options)}."
        )

    # ── Build head: [BOS] rubric_text [OPT_0] desc_0 [OPT_1] desc_1 ... [EOS] ──
    head_ids, marker_positions = _build_head(tok, rubric, options, head_max_len)

    # ── Encode context, truncating to remaining budget ──
    context_budget = max(0, max_len - len(head_ids) - 1)  # -1 for closing [SEP]
    ctx_ids = encode_with_lock(
        tok,
        context.replace(tok.mask_token or "[MASK]", " "),
        add_special_tokens=False,
    )["input_ids"]

    if truncate_context_left:
        ctx_ids = ctx_ids[max(0, len(ctx_ids) - context_budget):]
    else:
        ctx_ids = ctx_ids[:context_budget]

    # ── Assemble: [BOS] context [SEP] head... [EOS] ──
    # head already starts with [BOS]; we insert context after [BOS]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    sep = [tok.sep_token_id] if tok.sep_token_id is not None else []
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []

    # head_ids already contains [BOS] from _build_head; splice context in
    # Format: [BOS] ctx [SEP] rubric [OPT_0] desc_0 ... [OPT_k] desc_k [EOS]
    full_ids = bos + ctx_ids + sep + head_ids[len(bos):] + eos

    # Shift marker positions to account for the prepended context tokens
    ctx_offset = len(bos) + len(ctx_ids) + len(sep)
    shifted_markers = [m + ctx_offset - len(bos) for m in marker_positions]

    # Clamp to max_len
    full_ids = full_ids[:max_len]
    shifted_markers = [m for m in shifted_markers if m < max_len]

    return full_ids, shifted_markers


def _build_head(
    tok,
    rubric: str,
    options: List[str],
    head_max_len: int = 256,
) -> Tuple[List[int], List[int]]:
    """Build the rubric + option-marker section of the sequence.

    Returns (head_ids, marker_positions_within_head).
    marker_positions_within_head are 0-based indices into head_ids.
    """
    rubric_ids = encode_with_lock(
        tok, rubric, add_special_tokens=False
    )["input_ids"]

    # Encode each option: [OPT_i] <space> description
    opt_id_blocks: List[List[int]] = []
    for i, desc in enumerate(options):
        token_str = f"{_opt_token(i)} {desc}"
        ids = encode_with_lock(
            tok,
            token_str,
            add_special_tokens=False,
            truncation=True,
            max_length=64,
        )["input_ids"]
        opt_id_blocks.append(ids)

    # Budget the rubric to whatever is left after all option blocks
    opt_total = sum(len(b) for b in opt_id_blocks)
    rubric_budget = max(8, head_max_len - opt_total)
    rubric_ids = rubric_ids[:rubric_budget]

    # Assemble head (no [BOS]/[EOS] — those are added by encode_decision)
    head_ids = rubric_ids + []
    marker_positions: List[int] = []
    for block in opt_id_blocks:
        # The [OPT_i] token is always the first token in the block.
        # Record its position in the head before appending.
        marker_positions.append(len(head_ids))
        head_ids.extend(block)

    return head_ids, marker_positions
