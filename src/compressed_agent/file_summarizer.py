"""Deterministic summaries for structured benchmark files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Tuple
import logging
import re
from collections import Counter, OrderedDict
from functools import lru_cache
import math

from summa import keywords as textrank_keywords
from summa import summarizer as textrank_summarizer
from wordfreq import zipf_frequency

import spacy  # type: ignore[import-error]
from spacy.language import Language  # type: ignore[import-error]


MAX_SUMMARY_CHARS = 1500
MAX_FACT_LINES = 14
MAX_FACT_DEPTH = 6
MAX_FACT_CANDIDATES = 600
MAX_CHILDREN_PER_OBJECT = 120
MAX_LIST_SAMPLES = 3
MAX_NP_TEXT_CHARS = 8000
MAX_NOUN_PHRASES = 24
_SPACY_NLP: "Language | None | bool" = None

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def summarize_data_file(file_path: Path, raw_text: str) -> str:
    """Return a compact, human-friendly summary for structured files."""
    path = Path(file_path)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return _cap(
            f"{path.name} contains plain text ({len(raw_text)} chars). "
            "Here is the opening excerpt:\n"
            f"{raw_text[:400].strip()}"
        )

    structure_line = _describe_structure(path.name, payload)
    fact_sentences = _collect_payload_facts(payload)
    LOGGER.info(
        "Summarizer[%s]: collected %d facts across %d scopes",
        path.name,
        len(fact_sentences),
        len({fact.split(':', 1)[0] for fact in fact_sentences}),
    )
    numeric_highlights = _extract_numeric_highlights(payload)
    identity_facts = _select_headline_facts(fact_sentences)
    noun_phrase_facts, noun_phrase_terms = _extract_noun_phrase_facts(fact_sentences)
    key_phrases = _extract_keywords(fact_sentences)
    LOGGER.info(
        "Summarizer[%s]: keywords=%s noun_phrases=%d",
        path.name,
        key_phrases,
        len(noun_phrase_terms),
    )
    augmented_facts = fact_sentences + noun_phrase_facts
    key_phrase_set = set(key_phrases) | {term.lower() for term in noun_phrase_terms}
    salient_candidates = _select_salient_facts(augmented_facts, key_phrase_set)
    salient_facts = identity_facts + [
        fact
        for fact in salient_candidates
        if fact not in identity_facts and not fact.startswith("noun_phrase.")
    ]
    salient_facts = [fact for fact in salient_facts if not _is_media_path(fact)]

    lines: List[str] = [f"## Summary of `{path.name}`", structure_line]
    if key_phrases:
        lines.append(f"Keywords: {', '.join(key_phrases)}")
    if salient_facts:
        lines.append("Key facts:")
        lines.extend(f"- {fact}" for fact in salient_facts[:MAX_FACT_LINES])
    if numeric_highlights:
        lines.append("Numeric highlights:")
        lines.extend(f"- {fact}" for fact in numeric_highlights if not _is_media_path(fact))
    LOGGER.info(
        "Summarizer[%s]: salient=%d numeric=%d",
        path.name,
        len(salient_facts),
        len(numeric_highlights),
    )

    summary = "\n".join(line for line in lines if line).strip()
    return _cap(summary or f"{path.name} parsed successfully but no summary was generated.")


def _describe_structure(file_name: str, payload: Any) -> str:
    if isinstance(payload, dict):
        keys = list(payload.keys())
        preview = ", ".join(keys[:6])
        suffix = "..." if len(keys) > 6 else ""
        return f"Top-level structure: JSON object with {len(keys)} keys ({preview}{suffix})."
    if isinstance(payload, list):
        return f"Top-level structure: JSON list with {len(payload)} entries."
    return f"Top-level structure: JSON value of type {type(payload).__name__}."


def _collect_payload_facts(payload: Any) -> List[str]:
    facts: List[str] = []
    _walk_payload(payload, (), facts, 0)
    return facts


def _walk_payload(value: Any, path: tuple[str, ...], sink: List[str], depth: int) -> None:
    if len(sink) >= MAX_FACT_CANDIDATES or depth > MAX_FACT_DEPTH:
        return

    if _is_scalar(value):
        fact = _format_fact(path, value)
        if fact:
            sink.append(fact)
        return

    if isinstance(value, dict):
        keys = _prioritize_keys(value.keys())
        if keys and depth <= 1:
            sink.append(_format_object_fact(path, keys))
        if depth <= 2:
            scalar_children = [
                f"{key}={_format_scalar(child)}"
                for key, child in value.items()
                if _is_scalar(child) and _format_scalar(child)
            ][:5]
            if scalar_children:
                label = _path_to_label(path) or "root"
                sink.append(f"{label} fields: {', '.join(scalar_children)}")
        for key in keys[:MAX_CHILDREN_PER_OBJECT]:
            _walk_payload(value[key], path + (key,), sink, depth + 1)
        return

    if isinstance(value, list):
        if depth <= 1:
            sink.append(_format_list_fact(path, value))
        for idx, item in enumerate(value[:MAX_LIST_SAMPLES]):
            _walk_payload(item, path + (f"[{idx}]",), sink, depth + 1)
        return


def _format_fact(path: tuple[str, ...], value: Any) -> str:
    label = _path_to_label(path) or "value"
    formatted = _format_scalar(value)
    if not formatted:
        return ""
    if _looks_like_url(formatted):
        return ""
    ending = "" if formatted.endswith((".", "!", "?")) else "."
    return f"{label}: {formatted}{ending}"


def _format_object_fact(path: tuple[str, ...], keys: Iterable[str]) -> str:
    label = _path_to_label(path) or "root"
    key_list = list(keys)
    preview = ", ".join(key_list[:5])
    suffix = "..." if len(key_list) > 5 else ""
    return f"{label} object with {len(key_list)} keys ({preview}{suffix})."


def _format_list_fact(path: tuple[str, ...], values: List[Any]) -> str:
    label = _path_to_label(path) or "root"
    size = len(values)
    scalar_samples = [
        sample
        for sample in (_format_scalar(item) for item in values if _is_scalar(item))
        if sample
    ][:3]

    if scalar_samples:
        preview = ", ".join(scalar_samples)
        suffix = f" (examples: {preview})"
    else:
        dict_sample = next((item for item in values if isinstance(item, dict)), None)
        if dict_sample:
            kv_pairs = [
                f"{key}={_format_scalar(val)}"
                for key, val in dict_sample.items()
                if _is_scalar(val) and _format_scalar(val)
            ][:3]
            suffix = f" (sample fields: {'; '.join(kv_pairs)})" if kv_pairs else ""
        else:
            suffix = ""

    return f"{label} list with {size} entries{suffix}."


def _extract_keywords(facts: List[str]) -> List[str]:
    if not facts:
        return []
    corpus = " ".join(_split_fact(fact)[1] for fact in facts if _split_fact(fact)[1])
    if len(corpus.split()) < 10:
        return []
    try:
        keywords = textrank_keywords.keywords(corpus, words=40, split=True, scores=False)
    except (ValueError, IndexError):
        return []
    filtered = [kw.strip() for kw in keywords if _is_informative_keyword(kw)]
    return filtered[:6]


def _select_salient_facts(facts: List[str], keywords: set[str]) -> List[str]:
    if not facts:
        return []
    corpus = " ".join(facts)
    textrank_sentences = _run_textrank_summary(corpus)
    fallback = _rank_by_generic_salience(facts, keywords)
    LOGGER.info(
        "Salience: textrank=%d fallback=%d keywords=%d",
        len(textrank_sentences),
        len(fallback),
        len(keywords),
    )

    ordered: List[str] = []
    for sequence in (textrank_sentences, fallback):
        for fact in sequence:
            sentence = fact.strip()
            if not sentence or sentence in ordered:
                continue
            ordered.append(sentence)
    return _limit_redundancy(ordered, MAX_FACT_LINES)


def _run_textrank_summary(corpus: str) -> List[str]:
    if len(corpus.split()) < 30:
        return []
    try:
        sentences = textrank_summarizer.summarize(
            corpus,
            ratio=0.2,
            words=200,
            split=True,
        )
    except ValueError:
        return []
    return _dedupe_sequence(sentences)


def _rank_by_generic_salience(facts: List[str], keywords: set[str]) -> List[str]:
    tokenized = [_tokenize(fact) for fact in facts]
    label_token_counts, label_token_quality = _label_token_stats(facts)
    frequency = Counter(token for tokens in tokenized for token in tokens)

    scored = []
    for idx, (fact, tokens) in enumerate(zip(facts, tokenized)):
        if not tokens:
            continue
        avg_frequency = sum(frequency[token] for token in tokens) / len(tokens)
        rarity = min(_token_zipf(token) for token in tokens if token) if tokens else 10.0
        label, value = _split_fact(fact)
        depth = label.count(".")
        numeric_value = any(ch.isdigit() for ch in value)
        semantic_priority = _semantic_priority(label, value, label_token_counts, label_token_quality)
        value_quality = _value_quality(value)
        zero_penalty = 1 if numeric_value and _is_trivial_numeric(value) else 0
        short_penalty = 1 if len(value.strip()) < 4 else 0
        info_penalty = 0 if len(set(tokens)) > 1 else 1
        keyword_penalty = 0 if any(keyword in fact.lower() for keyword in keywords) else 1
        structure_penalty = sum(
            1
            for segment in label.split(".")
            if len(segment) <= 2 or segment.isdigit() or "image" in segment.lower()
        )
        metric_penalty = 0 if any(ch.isdigit() for ch in label) and any(ch.isalpha() for ch in label) else 1
        scored.append(
            (
                (
                    -value_quality,
                    semantic_priority,
                    keyword_penalty,
                    structure_penalty,
                    metric_penalty,
                    depth,
                    zero_penalty,
                    short_penalty,
                    info_penalty,
                    0 if numeric_value else 1,
                    rarity,
                    avg_frequency,
                    idx,
                ),
                fact,
            )
        )

    scored.sort(key=lambda item: item[0])
    return [fact for _, fact in scored]


def _dedupe_sequence(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        candidate = item.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        if not compact:
            return ""
        limit = 150
        return compact[:limit] + ("..." if len(compact) > limit else "")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _path_to_label(path: tuple[str, ...]) -> str:
    if not path:
        return ""
    parts: List[str] = []
    for segment in path:
        if segment.startswith("[") and parts:
            parts[-1] = parts[-1] + segment
        else:
            parts.append(segment)
    return ".".join(parts)


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)
_TAG_VALUE_PATTERN = re.compile(r"^[a-z]{2,3}:[^\s]+$", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def _label_token_stats(facts: List[str]) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    quality: Counter[str] = Counter()
    for fact in facts:
        label, _ = _split_fact(fact)
        tokens = set(_tokenize(label.lower().replace("_", " ")))
        if tokens:
            counts.update(tokens)
            value = _split_fact(fact)[1]
            vq = _value_quality(value)
            for token in tokens:
                quality[token] += vq
    return counts, quality


@lru_cache(maxsize=8192)
def _token_zipf(token: str) -> float:
    if not token:
        return 10.0
    score = zipf_frequency(token, "en")
    return score if score else 10.0


def _semantic_priority(
    label: str,
    value: str,
    token_counts: Counter[str],
    token_quality: Counter[str],
) -> float:
    label_lower = label.lower()
    tokens = _tokenize(label_lower.replace("_", " "))
    if tokens:
        rarity = min(_token_zipf(token) for token in tokens)
        avg_token_count = sum(token_counts.get(token, 0) for token in tokens) / len(tokens)
        avg_token_quality = sum(
            token_quality.get(token, 0.0) / max(1, token_counts.get(token, 1)) for token in tokens
        ) / len(tokens)
    else:
        rarity = 8.0
        avg_token_count = 1.0
        avg_token_quality = 0.0

    frequency_penalty = math.log1p(avg_token_count)
    depth_penalty = 0.5 * label_lower.count(".")
    list_penalty = 0.5 * label_lower.count("[")
    digit_bonus = -0.5 if any(ch.isdigit() for ch in label_lower) else 0.0
    mixed_value_bonus = (
        -0.5 if (any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value)) else 0.0
    )
    quality_bonus = max(0.0, avg_token_quality / 2.0)

    score = (
        rarity
        + frequency_penalty
        + depth_penalty
        + list_penalty
        + digit_bonus
        + mixed_value_bonus
        - quality_bonus
    )
    return max(0.0, score)


def _value_quality(value: str) -> float:
    if not value:
        return 0.0
    tokens = _tokenize(value)
    if not tokens:
        return 0.0
    token_count = len(tokens)
    unique_count = len(set(tokens))
    unique_ratio = unique_count / token_count
    avg_zipf = sum(_token_zipf(token) for token in tokens) / token_count
    rarity_bonus = max(0.0, 7.0 - avg_zipf)
    has_digit = any(ch.isdigit() for ch in value)
    has_alpha = any(ch.isalpha() for ch in value)
    combo_bonus = 1.0 if (has_digit and has_alpha) else 0.0
    comma_penalty = 0.4 * value.count(",")
    length_penalty = max(0.0, (len(value) - 60) / 20.0)
    short_penalty = 0.5 if token_count < 2 else 0.0
    return (unique_ratio * 4.0) + rarity_bonus + combo_bonus - comma_penalty - length_penalty - short_penalty


def _is_informative_keyword(keyword: str) -> bool:
    candidate = keyword.strip().lower()
    if len(candidate) < 3 or candidate.isdigit():
        return False
    if _token_zipf(candidate) >= 6.5:
        return False
    return True


def _looks_like_url(value: str) -> bool:
    return bool(_URL_PATTERN.search(value))


def _looks_like_taxonomy_tag(value: str) -> bool:
    return bool(_TAG_VALUE_PATTERN.match(value.strip()))


def _is_media_path(fact: str) -> bool:
    label = fact.split(":", 1)[0].lower()
    media_tokens = ("image", "thumb", "upload", "size")
    return any(token in label for token in media_tokens)


def _prioritize_keys(keys: Iterable[str]) -> List[str]:
    def score(key: str) -> tuple:
        tokens = _tokenize(key)
        rarity = min(_token_zipf(token) for token in tokens) if tokens else 10.0
        return (rarity, len(key), key)

    return sorted(keys, key=score)


def _split_fact(fact: str) -> tuple[str, str]:
    if ": " in fact:
        return fact.split(": ", 1)
    return fact, ""


def _is_trivial_numeric(value: str) -> bool:
    stripped = value.strip().rstrip(".")
    if not stripped:
        return True
    try:
        return float(stripped) == 0.0
    except ValueError:
        return False


def _scope_weight(fact: str) -> float:
    label, value = _split_fact(fact)
    tokens = _tokenize(value)
    digit_count = sum(ch.isdigit() for ch in value)
    non_zero_bonus = 2.0 if not _is_trivial_numeric(value) else 1.0
    token_bonus = 1.0 + len(set(tokens))
    digit_bonus = 1.0 + (digit_count > 0)
    depth_bonus = 2.0 if label.count(".") <= 1 else 1.0
    return non_zero_bonus * token_bonus * digit_bonus * depth_bonus


NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numeric_highlights(payload: Any, limit: int = 8) -> List[str]:
    candidates: List[tuple[tuple[str, ...], float, str]] = []

    def visit(value: Any, path: tuple[str, ...], depth: int) -> None:
        if depth > MAX_FACT_DEPTH:
            return
        if isinstance(value, (int, float)):
            if len(path) > 3:
                return
            candidates.append((path, float(value), str(value)))
            return
        if isinstance(value, str):
            number = _parse_numeric_fragment(value)
            if number is not None:
                if len(path) > 3:
                    return
                candidates.append((path, number, value.strip()))
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + (key,), depth + 1)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value[:MAX_LIST_SAMPLES]):
                visit(child, path + (f"[{idx}]",), depth + 1)

    visit(payload, (), 0)

    scored = []
    seen_values: set[str] = set()

    for path, numeric_value, original in candidates:
        if original in seen_values:
            continue
        seen_values.add(original)
        depth = len(path)
        scope = ".".join(path[:2]) if len(path) >= 2 else ".".join(path)
        scored.append(
            (
                1 if abs(numeric_value) > 1_000_000 else 0,
                -abs(numeric_value),
                depth,
                len(scope),
                scope,
                ".".join(path),
                original,
            )
        )

    scored.sort()
    result: List[str] = []
    used_scopes: set[str] = set()
    for _, _, _, _, scope, label, original in scored:
        if scope in used_scopes and len(used_scopes) < limit // 2:
            continue
        used_scopes.add(scope)
        result.append(f"{label}: {original}")
        if len(result) >= limit:
            break
    LOGGER.info("Numeric highlights selected=%d (candidates=%d)", len(result), len(candidates))
    return result


def _parse_numeric_fragment(value: str) -> float | None:
    match = NUMERIC_PATTERN.search(value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_noun_phrase_facts(facts: List[str]) -> Tuple[List[str], List[str]]:
    text_parts: List[str] = []
    consumed = 0
    for fact in facts:
        _, value = _split_fact(fact)
        if not value:
            continue
        text_parts.append(value)
        consumed += len(value)
        if consumed >= MAX_NP_TEXT_CHARS:
            break
    text = " ".join(text_parts)[:MAX_NP_TEXT_CHARS]
    phrases = _noun_phrases(text)
    seen: set[str] = set()
    filtered: List[str] = []
    for phrase in phrases:
        cleaned = phrase.strip()
        if len(cleaned) < 4 or len(cleaned) > 60:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(cleaned)
        if len(filtered) >= MAX_NOUN_PHRASES:
            break
    noun_phrase_facts = [f"noun_phrase.{idx}: {phrase}" for idx, phrase in enumerate(filtered)]
    return noun_phrase_facts, filtered


def _noun_phrases(text: str) -> List[str]:
    if not text:
        return []
    nlp = _get_spacy_nlp()
    if nlp and "parser" in nlp.pipe_names:
        doc = nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    return _fallback_noun_phrases(text)


def _fallback_noun_phrases(text: str) -> List[str]:
    pattern = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    matches = pattern.findall(text)
    return matches


def _get_spacy_nlp() -> "Language | None":
    global _SPACY_NLP
    if _SPACY_NLP is False:
        return None
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if spacy is None:
        _SPACY_NLP = False
        return None
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=("ner",))
    except OSError:
        try:
            from spacy.lang.en import English  # type: ignore[import-error]

            nlp = English()
            nlp.add_pipe("sentencizer")
            _SPACY_NLP = nlp
        except Exception:
            _SPACY_NLP = False
            return None
    return _SPACY_NLP


def _select_headline_facts(facts: List[str], limit: int = 6) -> List[str]:
    if not facts:
        return []

    label_token_counts, label_token_quality = _label_token_stats(facts)
    scored: List[tuple[tuple, str]] = []

    for fact in facts:
        label, value = _split_fact(fact)
        text = value.strip()
        if not text or len(text) < 4 or len(text) > 160:
            continue
        if _looks_like_url(text) or _looks_like_taxonomy_tag(text):
            continue
        word_tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        if len(word_tokens) < 2:
            continue
        digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
        if digit_ratio > 0.35:
            continue
        value_quality = _value_quality(text)
        if value_quality <= 0:
            continue
        semantic_priority = _semantic_priority(
            label,
            text,
            label_token_counts,
            label_token_quality,
        )
        depth = label.count(".")
        cap_ratio = sum(1 for token in word_tokens if token[0].isupper()) / len(word_tokens)
        score = (
            -value_quality,
            -cap_ratio,
            semantic_priority,
            depth,
            len(text),
            label,
        )
        scored.append((score, fact))

    scored.sort(key=lambda item: item[0])
    results: List[str] = []
    seen_scopes: set[str] = set()
    for _, fact in scored:
        label = fact.split(":", 1)[0]
        segments = label.split(".")
        scope_token = segments[1] if len(segments) > 1 else segments[0]
        scope_token = scope_token.split("[", 1)[0]
        scope_root = scope_token.split("_", 1)[0] if scope_token else scope_token
        if scope_root in seen_scopes:
            continue
        seen_scopes.add(scope_root)
        results.append(fact)
        if len(results) >= limit:
            break
    return results


def _limit_redundancy(facts: List[str], limit: int) -> List[str]:
    """Interleave scopes while prioritizing informative ones."""
    buckets: "OrderedDict[str, List[str]]" = OrderedDict()
    weights: dict[str, float] = {}
    for fact in facts:
        label = fact.split(":", 1)[0]
        scope = ".".join(label.split(".")[:2]) if "." in label else label
        buckets.setdefault(scope, []).append(fact)
        weights[scope] = weights.get(scope, 0.0) + _scope_weight(fact)

    ordered_scopes = sorted(buckets.keys(), key=lambda scope: (-weights.get(scope, 0.0), scope))
    result: List[str] = []

    while ordered_scopes and len(result) < limit:
        next_scopes: List[str] = []
        for scope in ordered_scopes:
            bucket = buckets.get(scope)
            if not bucket:
                continue
            result.append(bucket.pop(0))
            if bucket:
                next_scopes.append(scope)
            if len(result) >= limit:
                break
        ordered_scopes = next_scopes

    return result


def _cap(text: str) -> str:
    if len(text) <= MAX_SUMMARY_CHARS:
        return text
    return text[: MAX_SUMMARY_CHARS - 3].rstrip() + "..."

