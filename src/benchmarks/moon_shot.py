"""High-aggression benchmark runner focused on F1/token efficiency."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import json
import re

from pydantic import BaseModel, Field

from src.agent_run_summary import AgentRunSummary
import constants
from src.compressed_agent.file_summarizer import summarize_data_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_SENTENCE_EXTRACTIONS = 12
MIN_SENTENCE_TOKENS = 6
MAX_SENTENCE_TOKENS = 50


class SectionSpec(BaseModel):
    """Defines how to pull lines into a semantic section."""

    title: str = Field(..., description="Section heading.")
    keywords: Sequence[str] = Field(..., description="Case-insensitive substrings to match.")
    limit: int = Field(..., ge=1, description="Maximum lines to include.")


SECTION_SPECS: tuple[SectionSpec, ...] = (
    SectionSpec(title="Product Basics", keywords=("product", "brand", "countries", "serving"), limit=6),
    SectionSpec(title="Nutritional Information", keywords=("nutri", "energy", "fat", "protein", "carb", "fiber", "salt", "sugar"), limit=8),
    SectionSpec(title="Health & Ratings", keywords=("score", "grade", "rating", "nova", "ecoscore"), limit=6),
    SectionSpec(title="Key Ingredients", keywords=("ingredient", "recipe", "composition"), limit=6),
    SectionSpec(title="Allergens & Labels", keywords=("allergen", "label", "diet", "flag"), limit=6),
    SectionSpec(title="Packaging & Sustainability", keywords=("packaging", "eco", "footprint", "emission"), limit=6),
    SectionSpec(title="Data Quality", keywords=("quality", "completeness", "contributors", "last_edit", "entry"), limit=4),
)


@dataclass
class LineItem:
    text: str
    normalized: str
    used: bool = False


class ProductNarrativeBuilder:
    """Turns structured payloads into narrative sections."""

    def __init__(self, payload: Any):
        if isinstance(payload, dict):
            self.payload = payload
            product = payload.get("product")
            if isinstance(product, dict):
                self.product = product
            else:
                self.product = payload
        else:
            self.payload = None
            self.product = None

    def build_sections(self) -> List[str]:
        if not isinstance(self.product, dict):
            return []

        sections: List[str] = []

        def add_section(title: str, lines: List[str]) -> None:
            if lines:
                sections.append(f"### {title}")
                sections.extend(f"- {line}" for line in lines)

        add_section("Product Overview", self._overview_lines())
        add_section("Nutrition Snapshot", self._nutrition_lines())
        add_section("Health & Ratings", self._rating_lines())
        add_section("Ingredients & Allergens", self._ingredient_lines())
        add_section("Labels & Dietary Notes", self._label_lines())
        add_section("Packaging & Sustainability", self._packaging_lines())
        add_section("Data Quality", self._data_quality_lines())

        return sections

    # --- Section helpers -------------------------------------------------
    def _overview_lines(self) -> List[str]:
        name = self._text_field("product_name", "product_name_en", "generic_name")
        brand = self._text_field("brands")
        categories = self._format_join(self._list_field("categories_tags", "categories"), max_items=3)
        countries = self._format_join(self._list_field("countries_tags", "countries"), max_items=4)
        serving = self._text_field("serving_size")
        packaging = self._format_join(self._list_field("packaging", "packaging_tags"), max_items=3)

        lines: List[str] = []
        parts: List[str] = []
        if name:
            parts.append(name)
        if brand:
            parts.append(f"by {brand}")
        if categories:
            parts.append(f"({categories})")
        if parts:
            intro = " ".join(parts).strip()
            if countries:
                intro += f" sold in {countries}."
            else:
                intro += "."
            lines.append(intro)
        if serving or packaging:
            detail: List[str] = []
            if serving:
                detail.append(f"serving size {serving}")
            if packaging:
                detail.append(f"packaging: {packaging}")
            lines.append("; ".join(detail) + ".")
        return [line for line in lines if line]

    def _nutrition_lines(self) -> List[str]:
        nutriments = self.product.get("nutriments")
        if not isinstance(nutriments, dict):
            return []

        energy_kcal = self._format_quantity(nutriments.get("energy-kcal_100g"), "kcal")
        energy_kj = self._format_quantity(
            nutriments.get("energy-kj_100g") or nutriments.get("energy_100g"), "kJ"
        )
        macros: List[str] = []
        for label, key in [
            ("carbs", "carbohydrates_100g"),
            ("sugars", "sugars_100g"),
            ("fat", "fat_100g"),
            ("sat. fat", "saturated-fat_100g"),
            ("protein", "proteins_100g"),
            ("fiber", "fiber_100g"),
            ("salt", "salt_100g"),
        ]:
            value = self._format_quantity(nutriments.get(key), "g")
            if value:
                macros.append(f"{label} {value}")

        lines: List[str] = []
        energy_parts = [part for part in [energy_kcal, energy_kj] if part]
        if energy_parts or macros:
            head = "Per 100g: "
            pieces = energy_parts + macros
            head += " | ".join(pieces)
            lines.append(head)

        if not lines:
            return []
        return lines

    def _rating_lines(self) -> List[str]:
        lines: List[str] = []
        grade = self._text_field("nutriscore_grade", "nutrition_grade_fr")
        score = self.product.get("nutriscore_score")
        version = self.product.get("nutriscore_version")
        ecoscore = self.product.get("ecoscore_grade") or self._nested_text(
            self.product, ("ecoscore_data", "grade")
        )
        nova = self.product.get("nova_group")

        rating_bits: List[str] = []
        if grade:
            bit = f"Nutri-Score {grade.upper()}"
            if version:
                bit += f" ({version})"
            if score is not None:
                bit += f" score {score}"
            rating_bits.append(bit)
        if ecoscore:
            rating_bits.append(f"Eco-Score {str(ecoscore).upper()}")
        if nova:
            rating_bits.append(f"NOVA group {nova}")
        if rating_bits:
            lines.append(" | ".join(rating_bits))

        warnings = self._list_field("warnings", "additives_tags")
        if warnings:
            lines.append(f"Additives / warnings: {self._format_join(warnings)}.")

        return lines

    def _ingredient_lines(self) -> List[str]:
        ingredients_text = self._text_field("ingredients_text", "ingredients_text_with_allergens")
        ingredients = self._format_join(self._list_field("ingredients_tags"), max_items=6)
        allergens = self._format_join(self._list_field("allergens_tags", "allergens"), max_items=4)

        lines: List[str] = []
        if ingredients_text:
            lines.append(ingredients_text)
        elif ingredients:
            lines.append(f"Key ingredients: {ingredients}.")
        if allergens:
            lines.append(f"Declared allergens: {allergens}.")
        return lines

    def _label_lines(self) -> List[str]:
        labels = self._format_join(self._list_field("labels_tags", "labels"), max_items=6)
        diets = self._format_join(self._list_field("ingredients_analysis_tags"), max_items=6)
        suitability = self._format_join(self._list_field("traces_tags"), max_items=4)
        lines: List[str] = []
        if labels:
            lines.append(f"Labels / claims: {labels}.")
        if diets:
            lines.append(f"Dietary flags: {diets}.")
        if suitability:
            lines.append(f"Trace mentions: {suitability}.")
        return lines

    def _packaging_lines(self) -> List[str]:
        packaging = self._format_join(self._list_field("packaging", "packaging_tags"), max_items=4)
        recycling = self._format_join(
            self._list_field("packaging_recycling", "packaging_recycling_tags"),
            max_items=3,
        )
        ecoscore_data = self.product.get("ecoscore_data")
        lines: List[str] = []
        if packaging:
            lines.append(f"Packaging: {packaging}.")
        if recycling:
            lines.append(f"Recycling guidance: {recycling}.")
        if isinstance(ecoscore_data, dict):
            footprint = ecoscore_data.get("agribalyse", {}).get("co2_total")
            if footprint:
                lines.append(f"Estimated CO₂ footprint: {footprint} kg CO₂e.")
        return lines

    def _data_quality_lines(self) -> List[str]:
        quality = self._format_join(
            self._list_field("data_quality_info_tags", "data_quality_tags"),
            max_items=4,
        )
        completeness = self.product.get("completeness")
        last_edit = self._format_join(self._list_field("last_edit_dates_tags"), max_items=2)
        entry = self._format_join(self._list_field("entry_dates_tags"), max_items=2)
        contributors = self._format_join(self._list_field("informers_tags"), max_items=3)

        lines: List[str] = []
        if quality:
            lines.append(f"Quality flags: {quality}.")
        if completeness:
            lines.append(f"Reported completeness: {completeness}.")
        if last_edit:
            lines.append(f"Last updated: {last_edit}.")
        if entry:
            lines.append(f"Originally entered: {entry}.")
        if contributors:
            lines.append(f"Key contributors: {contributors}.")
        return lines

    # --- Utility helpers -------------------------------------------------
    def _text_field(self, *keys: str) -> str | None:
        for key in keys:
            value = self.product.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
            else:
                cleaned = str(value).strip() if value is not None else ""
            if cleaned:
                return cleaned
        return None

    def _list_field(self, *keys: str) -> List[str]:
        for key in keys:
            value = self.product.get(key)
            normalized = self._normalize_list(value)
            if normalized:
                return normalized
        return []

    def _nested_text(self, obj: dict, path: tuple[str, ...]) -> str | None:
        current: Any = obj
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        if isinstance(current, str):
            cleaned = current.strip()
        else:
            cleaned = str(current).strip() if current is not None else ""
        return cleaned or None

    @staticmethod
    def _normalize_list(value: Any) -> List[str]:
        def polish(token: str) -> str:
            if ":" in token and token.split(":", 1)[0].isalpha() and len(token.split(":", 1)[0]) <= 5:
                token = token.split(":", 1)[1]
            return token.replace("_", " ").strip()

        if isinstance(value, list):
            return [polish(str(item)) for item in value if str(item).strip()]
        if isinstance(value, str):
            parts = re.split(r"[;,|]", value)
            return [polish(part) for part in parts if part.strip()]
        return []

    @staticmethod
    def _format_join(items: List[str], max_items: int = 4) -> str:
        if not items:
            return ""
        display = items[:max_items]
        if len(items) > max_items:
            display.append("…")
        return ", ".join(display)

    @staticmethod
    def _format_quantity(value: Any, unit: str) -> str | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if abs(number) >= 100:
            formatted = f"{number:.0f}"
        elif abs(number) >= 10:
            formatted = f"{number:.1f}"
        else:
            formatted = f"{number:.2f}"
        formatted = formatted.rstrip("0").rstrip(".")
        return f"{formatted}{unit}"


class MoonShotComposer:
    """Orchestrates deterministic summarization with semantic post-processing."""

    def __init__(self, file_path: Path, raw_text: str):
        self.file_path = file_path
        self.raw_text = raw_text
        self.payload = self._load_payload(raw_text)
        self.narrative_builder = ProductNarrativeBuilder(self.payload)
        self.base_summary = summarize_data_file(file_path, raw_text)
        self.lines = self._extract_lines(self.base_summary)
        self.sentences = self._extract_sentences(self.base_summary)

    def compose(self) -> str:
        sections: List[str] = [f"## Moonshot Summary of `{self.file_path.name}`", self._headline()]

        structured = self.narrative_builder.build_sections()
        if structured:
            sections.extend(structured)
        else:
            sections.extend(self._keyword_sections())

        highlights = self._remaining_lines(limit=6)
        if highlights:
            sections.append("### Deterministic Highlights")
            sections.extend(f"- {line}" for line in highlights)

        return "\n".join(section for section in sections if section).strip()

    def _headline(self) -> str:
        metadata = self._extract_metadata_snapshot()
        if not metadata:
            return "This summary prioritizes factual breadth while minimizing token usage."
        return f"This summary balances coverage ({metadata}) while minimizing token usage."

    def _extract_metadata_snapshot(self) -> str:
        if isinstance(self.payload, dict):
            payload = self.payload
        else:
            try:
                payload = json.loads(self.raw_text)
            except json.JSONDecodeError:
                return ""

        top_keys = list(payload)[:5] if isinstance(payload, dict) else []
        if not top_keys:
            return ""
        return f"top-level keys: {', '.join(top_keys)}"

    @staticmethod
    def _extract_lines(summary_text: str) -> List[LineItem]:
        items: List[LineItem] = []
        for line in summary_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("##"):
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            lowered = stripped.lower()
            if lowered in {"key facts:", "numeric highlights:", "keywords:"}:
                continue
            if lowered.endswith(":") and len(lowered.split()) <= 3:
                continue
            normalized = stripped.lower()
            if not normalized:
                continue
            items.append(LineItem(text=stripped, normalized=normalized))
        return items

    def _collect_lines(self, keywords: Iterable[str], limit: int) -> List[str]:
        selected: List[str] = []
        for item in self.lines:
            if item.used:
                continue
            if any(keyword in item.normalized for keyword in keywords):
                selected.append(item.text)
                item.used = True
            if len(selected) >= limit:
                break
        return selected

    def _remaining_lines(self, limit: int) -> List[str]:
        leftovers = [item.text for item in self.lines if not item.used]
        return leftovers[:limit]

    def _keyword_sections(self) -> List[str]:
        sections: List[str] = []
        for spec in SECTION_SPECS:
            section_lines = self._collect_lines(spec.keywords, spec.limit)
            if section_lines:
                sections.append(f"### {spec.title}")
                sections.extend(f"- {line}" for line in section_lines)
        return sections

    @staticmethod
    def _load_payload(raw_text: str) -> Any:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return None

    def _extract_sentences(self, text: str) -> List[str]:
        sentences = _split_sentences(text)
        scored: List[tuple[float, str]] = []
        for sentence in sentences:
            score = _score_sentence(sentence)
            if score <= 0:
                continue
            scored.append((-score, sentence))
        scored.sort()
        unique_sentences: List[str] = []
        seen: set[str] = set()
        for _, sentence in scored:
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_sentences.append(sentence.strip())
            if len(unique_sentences) >= MAX_SENTENCE_EXTRACTIONS:
                break
        return unique_sentences


def run_moon_shot(task_override: str | None = None) -> AgentRunSummary:
    goal = (task_override or constants.task or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty; provide a summarization goal.")

    file_path = _extract_goal_file_path(goal)
    if not file_path:
        raise ValueError(f"Unable to determine file path from goal: {goal}")

    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Source file not found: {full_path}")

    raw_text = full_path.read_text(encoding="utf-8")
    composer = MoonShotComposer(full_path, raw_text)
    final_summary = composer.compose()

    usage = _estimate_usage(raw_text, final_summary)
    metadata = {
        "final_answer": final_summary,
        "source_file": str(full_path),
        "line_count": final_summary.count("\n") + 1,
    }

    return AgentRunSummary.from_usage(
        usage=usage,
        metadata=metadata,
    )


def _extract_goal_file_path(goal: str) -> Path | None:
    marker = "summarize file "
    goal_lower = goal.lower()
    if marker not in goal_lower:
        return None
    start_idx = goal_lower.index(marker) + len(marker)
    remainder = goal[start_idx:].strip()
    if not remainder:
        return None
    candidate = remainder.split()[0]
    return Path(candidate)


def _estimate_usage(source_text: str, summary_text: str) -> Dict[str, int]:
    def _token_count(text: str) -> int:
        return max(1, len(text.split()))

    input_tokens = _token_count(source_text)
    output_tokens = _token_count(summary_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sanitized = re.sub(r"^\s*[-•]+\s*", "", text, flags=re.MULTILINE)
    sanitized = sanitized.replace("###", ". ").replace("##", ". ")
    parts = re.split(r"(?<=[.!?])\s+", sanitized.strip())
    return [part.strip() for part in parts if part.strip()]


def _score_sentence(sentence: str) -> float:
    tokens = sentence.split()
    token_count = len(tokens)
    if token_count < MIN_SENTENCE_TOKENS or token_count > MAX_SENTENCE_TOKENS:
        return 0.0

    unique_token_ratio = len(set(token.lower() for token in tokens)) / token_count
    digit_bonus = 0.5 if any(ch.isdigit() for ch in sentence) else 0.0
    proper_noun_bonus = 0.1 * sum(
        1 for word in tokens if word[:1].isupper() and word[0].isalpha()
    )
    punctuation_penalty = 0.2 * sentence.count(";")

    return 1.0 + unique_token_ratio + digit_bonus + proper_noun_bonus - punctuation_penalty


if __name__ == "__main__":
    summary = run_moon_shot()
    print(summary.model_dump())

