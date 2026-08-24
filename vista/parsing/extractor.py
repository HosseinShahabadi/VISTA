"""Prompt parsing: attribute, spatial and numerical structure extraction.

Turns a raw prompt into the three structures the VISTA objectives consume:

  pairs            {adj, noun, adj_t5_ids, noun_t5_ids}
  spatial_pairs    {noun_a, relation, noun_b, axis, noun_a_side, *_t5_ids}
                   axis is 'x' (left/right), 'y' (top/bottom), or None for
                   depth relations such as "behind" / "in front of"
  numerical_pairs  {number, number_surface, noun, relation, *_t5_ids}

Strategy: try spaCy (transformer model) and fall back to a rule-based parser,
so the pipeline still runs when spaCy weights are unavailable. Character spans
are then mapped to T5 token ids against the same tokenizer the generator uses.
"""

import os
import re
import glob
import logging
from typing import Optional


# The spaCy transformer parser is pinned to CPU: the generator already owns the
# GPU, and loading en_core_web_trf onto it costs memory for no benefit. This
# line is why the module must be imported lazily, AFTER the generator has
# initialised CUDA — importing it at process start would hide the GPU from torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings
warnings.filterwarnings("ignore", message=".*W095.*")

logger = logging.getLogger(__name__)


# ── spaCy (optional) ────────────────────────────────────────────────────────
_nlp = None


def _try_load_spacy(weights_dir: Optional[str] = None) -> bool:
    global _nlp
    import spacy

    if weights_dir:
        pattern = os.path.join(
            weights_dir, "spacy", "en_core_web_trf", "en_core_web_trf-*"
        )
        matches = glob.glob(pattern)
        if matches:
            try:
                _nlp = spacy.load(matches[0])
                logger.info(f"[adj_extractor] spaCy loaded from {matches[0]}")
                return True
            except Exception as e:
                logger.warning(f"[adj_extractor] spaCy load failed from path: {e}")

    for model_name in ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm"):
        try:
            _nlp = spacy.load(model_name)
            logger.info(f"[adj_extractor] spaCy loaded: {model_name}")
            return True
        except Exception:
            continue

    logger.warning("[adj_extractor] spaCy unavailable — using rule-based fallback")
    return False


def _init_spacy(weights_dir: Optional[str] = None):
    if _nlp is None:
        _try_load_spacy(weights_dir)


# ── T5 tokenizer ─────────────────────────────────────────────────────────────
_t5_tok = None


def _init_t5_tokenizer(t5_path: str):
    global _t5_tok
    if _t5_tok is None:
        from transformers import AutoTokenizer
        _t5_tok = AutoTokenizer.from_pretrained(t5_path)
        logger.info(f"[adj_extractor] T5 tokenizer loaded from {t5_path}")


# ═══════════════════════════════════════════════════════════════════
#  NUMERICAL
# ═══════════════════════════════════════════════════════════════════


# ── Word-number tables ───────────────────────────────────────────────────────
 
_ONES: dict[str, int] = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
 
_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
 
_MULTIPLIERS: dict[str, int] = {
    "hundred": 100, "thousand": 1_000, "million": 1_000_000,
}
 
_ALL_NUMBER_WORDS: set[str] = set(_ONES) | set(_TENS) | set(_MULTIPLIERS) | {"and"}
 
# Fuzzy quantifiers → integer approximation (longest keys first for greedy match)
_FUZZY: dict[str, int] = {
    "half a dozen": 6,
    "a couple of":  2,
    "a pair of":    2,
    "a trio of":    3,
    "dozens of":    24,
    "hundreds of":  200,
    "a quartet":    4,
    "a couple":     2,
    "a dozen":      12,
    "a pair":       2,
    "a trio":       3,
    "a few":        3,
    "several":      4,
    "dozens":       24,
    "hundreds":     200,
    "many":         5,
    "some":         3,
}
 
# Common adjectives that appear between a number and its noun
_ADJ_WORDS: set[str] = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "gray", "grey", "black", "white", "golden", "silver", "cyan", "magenta",
    "teal", "navy", "maroon", "violet", "indigo", "beige", "cream", "turquoise",
    "big", "large", "small", "tiny", "huge", "giant", "little", "enormous",
    "tall", "short", "long", "wide", "narrow", "thick", "thin",
    "shiny", "glossy", "matte", "rough", "smooth", "fluffy", "hairy",
    "wooden", "metallic", "plastic", "rusty", "old", "new",
    "round", "square", "triangular", "circular", "oval", "rectangular", "flat", "curved",
    "bright", "dark", "light", "deep", "soft", "hard", "warm", "cool",
    "young", "elderly", "happy", "angry", "cute", "beautiful", "ugly",
    "pretty", "lovely", "handsome", "colorful", "striped", "spotted",
}
 

_SKIP_AFTER_COUNT: set[str] = {"of", "the", "a", "an", "more", "less", "few", "dozen", "couple", "pair", "trio", "quartet", "several", "some", "many", "score", "hundred", "thousand", "half"} | _ADJ_WORDS
 
_STOPWORDS: set[str] = {
    "and", "or", "with", "on", "in", "at", "by", "to", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "its", "their",
}

def extract_numerical_pairs(prompt: str, t5_path: Optional[str] = None, spacy_weights_dir: Optional[str] = None,) -> list[dict]:
    enc= None
    if t5_path:
        _init_t5_tokenizer(t5_path)
        enc= _t5_tok(prompt, return_offsets_mapping=True)
    _init_spacy(spacy_weights_dir)
    try:
        if _nlp is not None:
            pairs = _spacy_extract_numerical(prompt, enc)
        else:
            raise RuntimeError("spaCy not loaded")
    except Exception as e:
        logger.warning(f"[adj_extractor] spaCy numerical extract failed ({e}), using rule-based")
        pairs= _rule_based_extract_numerical(prompt)
        if enc is not None:
            pairs = _attach_t5_ids_numerical(pairs, prompt, enc)
        return _deduplicate_numerical(pairs)

    # rule-based fallback if spaCy found nothing (mirrors spatial behaviour)
    if not pairs:
        logger.info("[adj_extractor] spaCy found no numerical pairs, trying rule-based")
        pairs= _rule_based_extract_numerical(prompt)
        if enc is not None:
            pairs = _attach_t5_ids_numerical(pairs, prompt, enc)
    return _deduplicate_numerical(pairs)


def _spacy_extract_numerical(prompt: str, enc=None) -> list[dict]:
    doc = _nlp(prompt)
    pairs = []

    for token in doc:
        # Case 1: nummod — "three cats", "2 dogs"
        if token.dep_ == "nummod" and token.like_num:
            noun = token.head
            if noun.pos_ in ("NOUN", "PROPN"):
                pairs.append(_make_num_pair(
                    number=token.text,
                    noun=noun.text,
                    relation="count",
                    rule="nummod",
                    prompt=prompt,
                    enc=enc,
                    num_tok=token,
                    noun_tok=noun,
                ))

        # Case 2: compound numerics — "10 to 20 cats"
        if token.dep_ == "nummod" and token.head.dep_ == "nummod":
            # range — both tokens modify the same noun grandparent
            noun = token.head.head
            pairs.append(_make_num_pair(
                number=f"{token.head.text}-{token.text}",
                noun=noun.text,
                relation="range",
                rule="nummod_range",
                prompt=prompt,
                enc=enc,
                num_tok=token,
                noun_tok=noun,
            ))

    return pairs

def _t5_ids_for_span(token, prompt: str, enc) -> list[int]:
    """Map a spaCy token's char offsets to T5 token indices."""
    offsets = enc["offset_mapping"]
    char_start = token.idx
    char_end   = token.idx + len(token.text)
    ids = []
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if s < char_end and e > char_start:
            ids.append(i)
    return list(dict.fromkeys(ids))


def _words_to_int(text: str) -> str:
    """Convert number surface form to digit string if possible, else return as-is."""
    text_lower = text.lower().strip()
    # already a digit
    if re.fullmatch(r"\d+", text_lower):
        return text_lower
    # single word lookup
    if text_lower in _ONES:
        return str(_ONES[text_lower])
    if text_lower in _TENS:
        return str(_TENS[text_lower])
    # multi-word (e.g. "twenty one" from spaCy compound)
    tokens = text_lower.split()
    result = _parse_word_number(tokens)
    if result:
        return str(result[0])
    return text  # fallback: return as-is

# def _words_to_int(text: str) -> str | None:
#     text_lower = text.lower().strip()
#     if re.fullmatch(r"\d+", text_lower):
#         return text_lower
#     if text_lower in _ONES:
#         return str(_ONES[text_lower])
#     if text_lower in _TENS:
#         return str(_TENS[text_lower])
#     tokens = text_lower.split()
#     result = _parse_word_number(tokens)
#     if result:
#         return str(result[0])
#     return None  # ← was: return text


def _make_num_pair(
    number: str,
    noun: str,
    relation: str,
    rule: str,
    prompt: str,
    enc,
    num_tok,
    noun_tok,
) -> dict:
    pair = {
        "type": "numerical",
        "number": _words_to_int(number),
        "number_surface": number,
        "noun": noun,
        "relation": relation,       # "count" | "range" | "ordinal"
        "noun_t5_ids": [],
        "number_t5_ids": [],
        "rule": rule,
    }
    if enc is not None:
        pair["noun_t5_ids"] = _t5_ids_for_span(noun_tok, prompt, enc)
        pair["number_t5_ids"] = _t5_ids_for_span(num_tok, prompt, enc)
    return pair


# ── Helpers ──────────────────────────────────────────────────────────────────
 
def _deduplicate_numerical(pairs: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for p in pairs:
        key = (p["number"].lower(), p["noun"].lower(), p["relation"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique

def _normalise(text: str) -> str:
    """Replace hyphens between number words with spaces, e.g. 'twenty-one' → 'twenty one'."""
    def _replace(m: re.Match) -> str:
        left, right = m.group(1).lower(), m.group(2).lower()
        if (left in _TENS and right in _ONES) or (left in _ONES and right in _ONES):
            return f"{m.group(1)} {m.group(2)}"
        return m.group(0)
    return re.sub(r"([a-zA-Z]+)-([a-zA-Z]+)", _replace, text)
 
 
def _parse_word_number(tokens: list[str]) -> tuple[int, int] | None:
    """
    Parse a word-based number from the START of *tokens* (already normalised/lowercased).
    Returns (value, tokens_consumed) or None.
    """
    value= 0
    current= 0
    i= 0
 
    while i < len(tokens):
        word = tokens[i]
        if word in _ONES:
            current+= _ONES[word]
            i += 1
        elif word in _TENS:
            current += _TENS[word]
            i += 1
        elif word in _MULTIPLIERS:
            if current == 0:
                current = 1
            mult = _MULTIPLIERS[word]
            current *= mult
            if mult >= 1000:
                value += current
                current = 0
            i += 1
        elif word == "and" and i + 1 < len(tokens) and (
            tokens[i + 1] in _ONES or tokens[i + 1] in _TENS
        ):
            i+= 1  # skip "and" in "three hundred and twenty"
        else:
            break
 
    total= value+current
    return (total, i) if (total > 0 and i > 0) else None
 
 
def _find_noun(tokens: list[str], start: int) -> str | None:
    """First token from *start* that is a plausible noun (not skip/stopword/number-word)."""
    for tok in tokens[start:]:
        t = tok.lower()
        if (
            t not in _SKIP_AFTER_COUNT
            and t not in _STOPWORDS
            and t not in _ALL_NUMBER_WORDS
            and t.isalpha()
        ):
            return tok
    return None


def _rule_based_extract_numerical(prompt: str) -> list[dict]:
    results: list[dict] = []
    claimed_nouns: set[str] = set()

    # Normalise hyphens in number compounds ("twenty-one" → "twenty one")
    text = _normalise(prompt.strip())

    # Build a clean token list (preserving original casing)
    tokens_raw   = re.findall(r"[a-zA-Z][\w'-]*", text)
    tokens_lower = [t.lower() for t in tokens_raw]

    # ── Pass 1: digit numbers ─────────────────────────────────────────────
    # Find each integer in text, then scan forward (in tokens) for the noun.
    for m in re.finditer(r"\b(\d+)\b", text):
        count = int(m.group(1))
        after = re.findall(r"[a-zA-Z][\w'-]*", text[m.end():])
        noun = _find_noun(after, 0)
        if noun and noun.lower() not in _STOPWORDS:
            # results.append({"count": count, "noun": noun, "rule": "digit"})
            results.append({"type": "numerical", "number": str(count), "noun": noun, "relation": "count", "noun_t5_ids": [], "number_t5_ids": [], "rule": "digit"})
            claimed_nouns.add(noun.lower())

    # ── Pass 2: word numbers ──────────────────────────────────────────────
    i = 0
    while i < len(tokens_lower):
        tok = tokens_lower[i]
        if tok in _ONES or tok in _TENS:
            result = _parse_word_number(tokens_lower[i:])
            if result:
                value, consumed = result
                noun = _find_noun(tokens_raw, i + consumed)
                if noun and noun.lower() not in _STOPWORDS:
                    # results.append({"count": value, "noun": noun, "rule": "word_number"})
                    # results.append({"type": "numerical", "number": str(value), "noun": noun, "relation": "count", "noun_t5_ids": [], "number_t5_ids": [], "rule": "word_number"})
                    results.append({"type": "numerical", "number": str(value), "number_surface": " ".join(tokens_raw[i:i+consumed]), "noun": noun, "relation": "count", "noun_t5_ids": [], "number_t5_ids": [], "rule": "word_number"})
                    claimed_nouns.add(noun.lower())
                i += consumed
                continue
        i += 1

    # ── Pass 3: fuzzy quantifiers ─────────────────────────────────────────
    text_lower = text.lower()
    for phrase, count in _FUZZY.items():
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", text_lower):
            after = re.findall(r"[a-zA-Z][\w'-]*", text[m.end():])
            noun = _find_noun(after, 0)
            if noun and noun.lower() not in _STOPWORDS and noun.lower() not in claimed_nouns:
                # results.append({"count": count, "noun": noun, "rule": f"fuzzy({phrase!r})"})
                # results.append({"type": "numerical", "number": str(count), "noun": noun, "relation": "count", "noun_t5_ids": [], "number_t5_ids": [], "rule": f"fuzzy({phrase!r})"})
                results.append({"type": "numerical", "number": str(count), "number_surface": phrase, "noun": noun, "relation": "count", "noun_t5_ids": [], "number_t5_ids": [], "rule": f"fuzzy({phrase!r})"})
                claimed_nouns.add(noun.lower())

    # ── Pass 4: indefinite article + optional adjectives + noun → count=1 ─
    # Pattern: (a|an) [adj]* noun
    # We scan token-by-token for "a"/"an" then skip adjectives to find noun.
    # i = 0
    # while i < len(tokens_lower):
    #     if tokens_lower[i] in ("a", "an"):
    #         noun = _find_noun(tokens_raw, i + 1)
    #         if (
    #             noun
    #             and noun.lower() not in _STOPWORDS
    #             and noun.lower() not in _ALL_NUMBER_WORDS
    #             and noun.lower() not in claimed_nouns
    #         ):
    #             # results.append({"count": 1, "noun": noun, "rule": "article(a/an)"})
    #             # results.append({"type": "numerical", "number": "1", "noun": noun, "relation": "count", "noun_t5_ids": [], "number_t5_ids": [], "rule": "article(a/an)"})
    #             results.append({"type": "numerical", "number": "1", "number_surface": tokens_raw[i], "noun": noun, "relation": "count", "noun_t5_ids": [], "number_t5_ids": [], "rule": "article(a/an)"})
    #             claimed_nouns.add(noun.lower())
    #     i+= 1
    return results


# ═══════════════════════════════════════════════════════════════════
#  SPATIAL RELATION TABLE
# ═══════════════════════════════════════════════════════════════════

# Maps a canonical relation keyword → (axis, noun_a_side)
# axis=None means no reliable 2D direction (depth/ambiguous) → separation only
# noun_a_side = which side of the image noun_A should occupy
SPATIAL_RELATIONS = {
    # left/right (x-axis)
    "left":         ("x", "left"),
    "right":        ("x", "right"),
    # above/below (y-axis)
    "above":        ("y", "top"),
    "below":        ("y", "bottom"),
    "under":        ("y", "bottom"),
    "underneath":   ("y", "bottom"),
    "over":         ("y", "top"),
    "top":          ("y", "top"),
    "bottom":       ("y", "bottom"),
    "atop":         ("y", "top"),
    # depth (3D — separation only, no 2D axis)
    "behind":       (None, None),
    "in front of":  (None, None),
    "front":        (None, None),
    "hidden by":    (None, None),
    "hidden":       (None, None),
    # proximity — separation only
    "side":         (None, None),
    "next to":      (None, None),
    "beside":       (None, None),
    "near":         (None, None),
    "alongside":    (None, None),
    "adjacent":     (None, None),
}

# Multi-word phrases to check before single words (order matters)
_MULTI_WORD_RELATIONS = ["in front of", "next to", "hidden by"]

# Prepositions/connectors that introduce NOUN_B after the relation
_BRIDGE_WORDS = {"of", "to", "the", "a", "an"}

# Side → (start_col_frac, end_col_frac) or (start_row_frac, end_row_frac)
# Used by the VISTA spatial objectives to build half-field masks
SIDE_TO_HALF = {
    "left":   (0.0, 0.5),
    "right":  (0.5, 1.0),
    "top":    (0.0, 0.5),
    "bottom": (0.5, 1.0),
}


# ═══════════════════════════════════════════════════════════════════
#  ADJ-NOUN EXTRACTION  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════

# _COLORS = {
#     "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
#     "gray", "grey", "black", "white", "golden", "silver", "cyan", "magenta",
#     "teal", "navy", "maroon", "violet", "indigo", "beige", "cream", "turquoise",
# }
# _SIZES = {
#     "big", "large", "small", "tiny", "huge", "giant", "little", "enormous",
#     "tall", "short", "long", "wide", "narrow", "thick", "thin",
# }
# _TEXTURES = {
#     "shiny", "glossy", "matte", "rough", "smooth", "fluffy", "hairy",
#     "wooden", "metallic", "plastic", "glass", "stone", "rusty", "old", "new",
# }
# _SHAPES = {
#     "round", "square", "triangular", "circular", "oval", "rectangular", "flat", "curved",
# }
# _OTHER_ADJ = {
#     "bright", "dark", "light", "deep", "soft", "hard", "warm", "cool",
#     "young", "elderly", "happy", "angry", "cute", "beautiful", "ugly",
#     "pretty", "lovely", "handsome", "colorful", "striped", "spotted",
# }


_COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "gray", "grey", "black", "white", "golden", "silver", "cyan", "magenta",
    "teal", "navy", "maroon", "violet", "indigo", "beige", "cream", "turquoise",
    # added: missing from val files
    "gold", "tan", "stainless",
}
_SIZES = {
    "big", "large", "small", "tiny", "huge", "giant", "little", "enormous",
    "tall", "short", "long", "wide", "narrow", "thick", "thin",
    # added
    "plump", "towering", "bulky",
}
_TEXTURES = {
    "shiny", "glossy", "matte", "rough", "smooth", "fluffy", "hairy",
    "wooden", "metallic", "plastic", "glass", "stone", "rusty", "old", "new",
    # added: texture_val and complex_val materials/surface qualities
    "leather", "rubber", "fabric",
    "fuzzy", "silky", "velvet", "polished", "sleek", "sparkling",
    "crispy", "crunchy", "gooey", "greasy",
    "rugged", "weathered", "gnarled", "tiled", "hardcover",
}
_SHAPES = {
    "round", "square", "triangular", "circular", "oval", "rectangular", "flat", "curved",
    # added: shape_val.txt uses these extensively
    "spherical", "cubic", "cylindrical", "pyramidal", "conical", "pentagonal",
    "oblong", "teardrop", "diamond", "crescent", "hexagonal",
    "angular", "rounded",
}
_OTHER_ADJ = {
    "bright", "dark", "light", "deep", "soft", "hard", "warm", "cool",
    "young", "elderly", "happy", "angry", "cute", "beautiful", "ugly",
    "pretty", "lovely", "handsome", "colorful", "striped", "spotted",
    # added: complex_val descriptive adjectives
    "sharp", "crisp", "cozy", "elegant", "vibrant", "bold",
    "delicate", "intricate", "ancient", "modern", "vintage", "ornate", "plain",
    "sweet", "sour", "juicy", "ripe", "fresh", "fragrant", "pungent", "savory",
    "cold", "hot", "icy", "frozen", "wet", "dry", "damp", "chilled", "chilly",
    "heavy", "solid", "clean", "dull", "dim", "glowing", "shimmering",
    "bumpy", "spiky", "jagged", "sticky", "slippery",
    "lush", "rich", "quiet", "translucent", "murky",
    "thorny", "sturdy", "tender", "sizzling",
    "graceful", "fluttering", "distant", "reedy",
    "diamond-studded"
}


ALL_ADJ = _COLORS | _SIZES | _TEXTURES | _SHAPES | _OTHER_ADJ


def _rule_based_extract_adj(prompt: str) -> list[dict]:
    tokens   = re.findall(r"[a-zA-Z'-]+", prompt.lower())
    original = re.findall(r"[a-zA-Z'-]+", prompt)
    pairs = []
    i = 0
    not_nouns= ["a", "an", "the", "and", "or", "with", "on", "near", "next", "to", "of", "in", "at", "by", "is", "are"]
    while i < len(tokens):
        tok = tokens[i]
        if tok in not_nouns:
            i += 1
            continue
        adjs = []
        j = i
        while j < len(tokens) and tokens[j] in ALL_ADJ:
            if adjs and tokens[j]=="orange":
                break
            adjs.append((j, tokens[j], original[j]))
            j += 1  
        
        if adjs and j < len(tokens) and (tokens[j] not in not_nouns) and ((tokens[j] not in ALL_ADJ) or  (tokens[j]=="orange")):
            noun_idx  = j
            noun_word = original[noun_idx]
            for adj_idx, adj_lower, adj_orig in adjs:
                pairs.append({
                    "adj":          adj_orig,
                    "noun":         noun_word,
                    "adj_t5_ids":   [],
                    "noun_t5_ids":  [],
                    "adj_spacy_i":  adj_idx,
                    "noun_spacy_i": noun_idx,
                    "rule":         "rule_based",
                })
            i = j + 1
        else:
            i += 1
    return pairs


def _spacy_extract_adj(prompt: str, enc) -> list[dict]:
    doc   = _nlp(prompt)
    pairs = []
    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ in ("NOUN", "PROPN"):
            pairs.append(_make_adj_pair(token, token.head, enc, "amod"))
        elif token.dep_ in ("acomp", "attr") and token.pos_ == "ADJ":
            subj = _find_subj(token.head)
            if subj is None and token.head.dep_ == "relcl":
                subj = token.head.head
            if subj is not None:
                pairs.append(_make_adj_pair(token, subj, enc, "predicative"))
        elif token.dep_ == "compound" and token.pos_ == "ADJ":
            pairs.append(_make_adj_pair(token, token.head, enc, "compound"))
        elif token.dep_ == "xcomp" and token.pos_ == "ADJ":
            subj = _find_subj(token.head)
            if subj is not None:
                pairs.append(_make_adj_pair(token, subj, enc, "xcomp"))
    return pairs


def _find_subj(verb_tok):
    for child in verb_tok.children:
        if child.dep_ in ("nsubj", "nsubjpass") and child.pos_ in ("NOUN", "PROPN"):
            return child
    return None


def _make_adj_pair(adj_tok, noun_tok, enc, rule: str) -> dict:
    return {
        "adj":          adj_tok.text,
        "noun":         noun_tok.text,
        "adj_t5_ids":   _char_to_t5_ids(adj_tok.idx, adj_tok.idx + len(adj_tok.text), enc) if enc else [],
        "noun_t5_ids":  _char_to_t5_ids(noun_tok.idx, noun_tok.idx + len(noun_tok.text), enc) if enc else [],
        "adj_spacy_i":  adj_tok.i,
        "noun_spacy_i": noun_tok.i,
        "rule":         rule,
    }


# ═══════════════════════════════════════════════════════════════════
#  SPATIAL PAIR EXTRACTION  (new)
# ═══════════════════════════════════════════════════════════════════

def _rule_based_extract_spatial(prompt: str) -> list[dict]:
    """
    Rule-based spatial extractor.
    Finds patterns: NOUN_A ... RELATION ... NOUN_B
    Works on lowercased token list with multi-word relation matching.

    Returns list of spatial pair dicts (T5 ids left empty, filled later).
    """
    text_lower = prompt.lower()
    tokens     = re.findall(r"[a-zA-Z'-]+", text_lower)
    original   = re.findall(r"[a-zA-Z'-]+", prompt)

    skip = {"a", "an", "the", "and", "or", "with", "at", "by", "is", "are",
            "on", "to", "of", "in", "its", "their"}

    def is_likely_noun(tok: str) -> bool:
        return tok not in skip and tok not in ALL_ADJ and tok not in _flat_relation_words()

    def _flat_relation_words():
        words = set()
        for key in SPATIAL_RELATIONS:
            words.update(key.split())
        return words

    pairs = []

    # Find which token indices are part of a relation keyword
    relation_spans = []   # list of (start_idx, end_idx_excl, canonical_key)

    # Check multi-word first, then single-word (greedy, left to right)
    for phrase in _MULTI_WORD_RELATIONS + [k for k in SPATIAL_RELATIONS if k not in _MULTI_WORD_RELATIONS]:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i+n] == phrase_tokens:
                # Check not already covered
                if not any(i < end and i + n > start for start, end, _ in relation_spans):
                    relation_spans.append((i, i + n, phrase))

    relation_spans.sort(key=lambda x: x[0])

    for rel_start, rel_end, rel_key in relation_spans:
        axis, noun_a_side = SPATIAL_RELATIONS[rel_key]

        # --- find NOUN_A: last noun-like token BEFORE rel_start ---
        noun_a_idx = None
        for i in range(rel_start - 1, -1, -1):
            if is_likely_noun(tokens[i]):
                noun_a_idx = i
                break

        # --- find NOUN_B: first noun-like token AFTER rel_end ---
        # skip bridge words ("of", "the", "a", ...)
        noun_b_idx = None
        for i in range(rel_end, len(tokens)):
            if tokens[i] in _BRIDGE_WORDS:
                continue
            if is_likely_noun(tokens[i]):
                noun_b_idx = i
                break

        if noun_a_idx is None or noun_b_idx is None:
            continue

        pairs.append({
            "type":           "spatial",
            "noun_a":         original[noun_a_idx],
            "noun_b":         original[noun_b_idx],
            "relation":       rel_key,
            "axis":           axis,
            "noun_a_side":    noun_a_side,
            "noun_a_t5_ids":  [],
            "noun_b_t5_ids":  [],
            "noun_a_rule_i":  noun_a_idx,
            "noun_b_rule_i":  noun_b_idx,
            "rule":           "rule_based_spatial",
        })

    return pairs


def _spacy_extract_spatial(prompt: str, enc) -> list[dict]:
    """
    spaCy-based spatial extractor.
    Looks for prepositional phrases (prep) whose head is a noun.

    Pattern: NOUN_A --(prep)--> PREP_TOKEN, PREP_TOKEN --(pobj)--> NOUN_B
    The prep token text (or its subtree text) maps to a SPATIAL_RELATIONS key.
    """
    doc   = _nlp(prompt)
    pairs = []

    for token in doc:
        # We want: noun --prep--> preposition token
        if token.dep_ != "prep":
            continue

        head = token.head
        if head.pos_ not in ("NOUN", "PROPN"):
            continue

        # Get the pobj (object of the preposition) = NOUN_B
        pobj = None
        for child in token.children:
            if child.dep_ == "pobj" and child.pos_ in ("NOUN", "PROPN"):
                pobj = child
                break
        if pobj is None:
            continue

        # Build the relation phrase from the prep subtree (handles "in front of")
        rel_phrase = _prep_subtree_text(token, pobj)
        rel_key    = _match_relation(rel_phrase)
        if rel_key is None:
            continue

        axis, noun_a_side = SPATIAL_RELATIONS[rel_key]

        pairs.append({
            "type":           "spatial",
            "noun_a":         head.text,
            "noun_b":         pobj.text,
            "relation":       rel_key,
            "axis":           axis,
            "noun_a_side":    noun_a_side,
            "noun_a_t5_ids":  _char_to_t5_ids(head.idx,  head.idx  + len(head.text),  enc) if enc else [],
            "noun_b_t5_ids":  _char_to_t5_ids(pobj.idx,  pobj.idx  + len(pobj.text),  enc) if enc else [],
            "noun_a_spacy_i": head.i,
            "noun_b_spacy_i": pobj.i,
            "rule":           "spacy_spatial",
        })

    return pairs


def _prep_subtree_text(prep_tok, exclude_tok) -> str:
    """
    Collect the text of a prep token and its children (excluding the pobj subtree).
    e.g. for "in front of", prep_tok="in", children include "front" and "of".
    """
    parts = [prep_tok.text]
    for child in prep_tok.children:
        if child != exclude_tok and child.dep_ != "pobj":
            parts.append(child.text)
    return " ".join(parts).lower()


def _match_relation(phrase: str) -> Optional[str]:
    """Return canonical relation key if phrase matches any SPATIAL_RELATIONS key."""
    phrase = phrase.lower().strip()
    # exact match first
    if phrase in SPATIAL_RELATIONS:
        return phrase
    # check if any key is contained in the phrase
    for key in _MULTI_WORD_RELATIONS:
        if key in phrase:
            return key
    for key in SPATIAL_RELATIONS:
        if key in phrase.split():
            return key
    return None


# ═══════════════════════════════════════════════════════════════════
#  T5 TOKEN ID HELPERS
# ═══════════════════════════════════════════════════════════════════

def _char_to_t5_ids(char_start: int, char_end: int, enc) -> list[int]:
    ids = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s == 0 and e == 0:
            continue
        if s < char_end and e > char_start:
            ids.append(i)
    return ids


def _attach_t5_ids_adj(pairs: list[dict], prompt: str, enc) -> list[dict]:
    """Fill T5 ids for rule-based adj-noun pairs by exact word match."""
    offsets = enc["offset_mapping"]

    def word_to_ids(word: str) -> list[int]:
        word_lower = word.lower()
        ids = []
        char_start = prompt.lower().find(word_lower)
        if char_start == -1:
            return []
        char_end = char_start + len(word_lower)
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s < char_end and e > char_start:
                ids.append(i)
        return list(dict.fromkeys(ids))

    for pair in pairs:
        if not pair["adj_t5_ids"]:
            pair["adj_t5_ids"]  = word_to_ids(pair["adj"])
        if not pair["noun_t5_ids"]:
            pair["noun_t5_ids"] = word_to_ids(pair["noun"])
    return pairs


def _attach_t5_ids_spatial(pairs: list[dict], prompt: str, enc) -> list[dict]:
    """Fill T5 ids for rule-based spatial pairs by exact word match."""
    offsets = enc["offset_mapping"]

    def word_to_ids(word: str) -> list[int]:
        word_lower = word.lower()
        char_start = prompt.lower().find(word_lower)
        if char_start == -1:
            return []
        char_end = char_start + len(word_lower)
        ids = []
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s < char_end and e > char_start:
                ids.append(i)
        return list(dict.fromkeys(ids))

    for pair in pairs:
        if not pair["noun_a_t5_ids"]:
            pair["noun_a_t5_ids"] = word_to_ids(pair["noun_a"])
        if not pair["noun_b_t5_ids"]:
            pair["noun_b_t5_ids"] = word_to_ids(pair["noun_b"])
    return pairs

def _attach_t5_ids_numerical(pairs: list[dict], prompt: str, enc) -> list[dict]:
    """Fill T5 ids for rule-based numerical pairs by exact word match."""
    offsets = enc["offset_mapping"]

    def word_to_ids(word: str) -> list[int]:
        word_lower = word.lower()
        char_start = prompt.lower().find(word_lower)
        if char_start == -1:
            return []
        char_end = char_start + len(word_lower)
        ids = []
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s < char_end and e > char_start:
                ids.append(i)
        return list(dict.fromkeys(ids))

    # for pair in pairs:
    #     if not pair["number_t5_ids"]:
    #         pair["number_t5_ids"] = word_to_ids(pair["number"])
    #     if not pair["noun_t5_ids"]:
    #         pair["noun_t5_ids"]   = word_to_ids(pair["noun"])
            
    for pair in pairs:
        if not pair["number_t5_ids"]:
            surface = pair.get("number_surface") or pair["number"]
            pair["number_t5_ids"] = word_to_ids(surface)
        if not pair["noun_t5_ids"]:
            pair["noun_t5_ids"] = word_to_ids(pair["noun"])
        
    return pairs


def _deduplicate_adj(pairs: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for p in pairs:
        key = (p["adj"].lower(), p["noun"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _deduplicate_spatial(pairs: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for p in pairs:
        key = (p["noun_a"].lower(), p["noun_b"].lower(), p["relation"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API — EXTRACTORS
# ═══════════════════════════════════════════════════════════════════


def extract_pairs(
    prompt: str,
    t5_path: Optional[str] = None,
    spacy_weights_dir: Optional[str] = None,
) -> list[dict]:
    """Extract adj-noun pairs by merging spaCy and rule-based results (deduplicated)."""
    enc = None
    if t5_path:
        _init_t5_tokenizer(t5_path)
        enc = _t5_tok(prompt, return_offsets_mapping=True)

    _init_spacy(spacy_weights_dir)

    spacy_pairs = []
    try:
        if _nlp is not None:
            spacy_pairs = _spacy_extract_adj(prompt, enc)
        else:
            raise RuntimeError("spaCy not loaded")
    except Exception as e:
        logger.warning(f"[adj_extractor] spaCy adj extract failed ({e})")
        spacy_pairs = []

    rule_pairs = _rule_based_extract_adj(prompt)
    if enc is not None:
        rule_pairs = _attach_t5_ids_adj(rule_pairs, prompt, enc)

    merged = spacy_pairs + rule_pairs
    return _deduplicate_adj(merged)

def extract_spatial_pairs(
    prompt: str,
    t5_path: Optional[str] = None,
    spacy_weights_dir: Optional[str] = None,
) -> list[dict]:
    """
    Extract spatial noun-relation-noun pairs from prompt.

    Returns list of dicts with keys:
        type          : "spatial"
        noun_a        : str  — the subject noun (e.g. "cat")
        noun_b        : str  — the object noun  (e.g. "dog")
        relation      : str  — canonical relation key (e.g. "left")
        axis          : "x" | "y" | None
        noun_a_side   : "left"|"right"|"top"|"bottom" | None
        noun_a_t5_ids : list[int]
        noun_b_t5_ids : list[int]
        rule          : str
    """
    enc = None
    if t5_path:
        _init_t5_tokenizer(t5_path)
        enc = _t5_tok(prompt, return_offsets_mapping=True)

    _init_spacy(spacy_weights_dir)
    try:
        if _nlp is not None:
            pairs = _spacy_extract_spatial(prompt, enc)
        else:
            raise RuntimeError("spaCy not loaded")
    except Exception as e:
        logger.warning(f"[adj_extractor] spaCy spatial extract failed ({e}), using rule-based")
        pairs = _rule_based_extract_spatial(prompt)
        if enc is not None:
            pairs = _attach_t5_ids_spatial(pairs, prompt, enc)

    # rule-based fallback if spaCy found nothing
    if not pairs:
        logger.info("[adj_extractor] spaCy found no spatial pairs, trying rule-based")
        pairs = _rule_based_extract_spatial(prompt)
        if enc is not None:
            pairs = _attach_t5_ids_spatial(pairs, prompt, enc)

    return _deduplicate_spatial(pairs)


def resolve_token_ids(pairs: list[dict], prompt: str, t5_path: str) -> list[dict]:
    """Fill empty T5 ids in adj-noun pairs via exact matching. Unchanged from v1."""
    _init_t5_tokenizer(t5_path)
    tokens  = _t5_tok(prompt, return_tensors='pt').input_ids[0]
    decoded = _t5_tok.convert_ids_to_tokens(tokens)

    def exact_match(word: str) -> list[int]:
        w = word.lower()
        return [i for i, tok in enumerate(decoded)
                if tok.lstrip('▁').lower() == w]

    for pair in pairs:
        if not pair.get('adj_t5_ids'):
            pair['adj_t5_ids']  = exact_match(pair['adj'])
        if not pair.get('noun_t5_ids'):
            pair['noun_t5_ids'] = exact_match(pair['noun'])
    return pairs


# ═══════════════════════════════════════════════════════════════════
#  CONVENIENCE
# ═══════════════════════════════════════════════════════════════════

def get_nouns_and_adjs(prompt: str, **kwargs) -> tuple[list[str], list[str]]:
    pairs = extract_pairs(prompt, **kwargs)
    nouns = list(dict.fromkeys(p["noun"] for p in pairs))
    adjs  = list(dict.fromkeys(p["adj"]  for p in pairs))
    return nouns, adjs


def decode_ids(ids: list[int], prompt: str, t5_path: str) -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(t5_path)
    enc = tok(prompt, return_offsets_mapping=True)
    return [enc.tokens()[i] for i in ids if i < len(enc.tokens())]
