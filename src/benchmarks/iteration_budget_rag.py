from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient, TokenUsage

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Minimal, task-specific system prompts (no roleplay, no verbosity)
SYSTEM_COMPRESS = "Compress the provided snippets into a compact, reusable fact set. No explanations."
SYSTEM_ANSWER = "Answer strictly and concisely. If unknown, say 'unknown'."

# Simple token estimator (words * 1.3 + 1)
def estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * 1.3) + 1)


class Slice(BaseModel):
    doc_id: str
    content: str
    score: float
    token_estimate: int


class CompressionResult(BaseModel):
    summary: str
    summary_tokens_estimate: int


class QAResult(BaseModel):
    answer: str
    confidence: str  # "high" | "low"
    stage: str       # "compressed" | "escalated"


@dataclass(frozen=True)
class Budgets:
    # Hard caps
    total_token_cap: int = 900
    # Per-call output caps
    compress_output: int = 140
    answer_low_output: int = 60
    answer_high_output: int = 110
    # Retrieval sizes
    topk_compress: int = 6
    topk_escalate: int = 12


def run_iteration_budget_rag(task_override: str | None = None) -> AgentRunSummary:
    """
    Budgeted RAG:
    - Cheap lexical retrieval → small top-k
    - Compress into reusable facts under strict token budget
    - Low-token answer on compressed view
    - Escalate once to larger context only if low confidence, respecting total token cap
    """
    goal = _require_goal(task_override or constants.task)
    file_path, question = _extract_file_and_question(goal)
    view = _load_view(file_path, question)
    budgets = Budgets()

    client = BedrockClient(model_id=constants.MODEL_ID)

    # 1) Retrieval (cheap lexical, no LLM)
    slices = list(_retrieve_slices(view, budgets.topk_compress))

    # 2) Compression (LLM, small budget)
    compression = _compress_slices(client, view, slices, budgets.compress_output)

    # 3) Low-cost reasoning using compressed view
    low_result = _answer_with_compressed(client, view, compression, budgets.answer_low_output)

    result = low_result
    # 4) Escalate once if low confidence and we have budget headroom
    if result.confidence == "low" and _has_budget_headroom(client.get_total_usage(), budgets, reserve=200):
        escalated_slices = list(_retrieve_slices(view, budgets.topk_escalate))
        result = _answer_with_escalation(
            client, view, compression, escalated_slices, budgets.answer_high_output
        )

    total_usage = client.get_total_usage().model_dump()
    metadata = {
        "final_answer": result.answer.strip(),
        "confidence": result.confidence,
        "stage": result.stage,
        "source_file": str(file_path),
        "question": question,
        "retrieval_topk_initial": budgets.topk_compress,
        "retrieval_topk_escalate": budgets.topk_escalate,
        "compression_tokens_estimate": compression.summary_tokens_estimate,
        "token_cap": budgets.total_token_cap,
    }
    return AgentRunSummary.from_usage(usage=total_usage, metadata=metadata)


def _require_goal(goal: str) -> str:
    normalized = goal.strip()
    if not normalized:
        raise ValueError("Task required")
    return normalized


def _extract_file_and_question(goal: str) -> tuple[Path, str]:
    """
    Accepts:
      - 'summarize file data/foo.json'
      - 'answer: <question> file data/foo.json'
      - '<any question> file data/foo.json'
    """
    lowered = goal.lower()
    marker = "file "
    if marker not in lowered:
        raise ValueError("Goal must include 'file <path>'")
    idx = lowered.index(marker) + len(marker)
    remainder = goal[idx:].strip()
    if not remainder:
        raise ValueError("No file path provided")
    # First token after 'file ' is the path
    path_token, *maybe_question_tokens = remainder.split()
    file_path = Path(path_token)

    # Question = everything before 'file <path>' or explicit 'answer:' prefix
    prefix = goal[: lowered.index(marker)].strip()
    question = prefix
    if not question:
        # Default to summarization if no question given
        question = f"Summarize the key facts of {file_path.name}"
    question = re.sub(r"^\s*answer:\s*", "", question, flags=re.IGNORECASE).strip()
    if not question:
        question = f"Summarize the key facts of {file_path.name}"
    return file_path, question


class DocumentView(BaseModel):
    doc_id: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)
    question: str


def _load_view(path: Path, question: str) -> DocumentView:
    full_path = (PROJECT_ROOT / path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Missing: {full_path}")
    text = full_path.read_text(encoding="utf-8")
    payload = _try_parse_json(text)
    return DocumentView(
        doc_id=full_path.name,
        path=str(path),
        raw_text=text,
        payload=payload,
        word_count=len(text.split()),
        question=question,
    )


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _retrieve_slices(view: DocumentView, top_k: int) -> Iterable[Slice]:
    """Cheap lexical retrieval using overlap scoring."""
    query_terms = _norm_terms(view.question)
    candidates: list[Slice] = []
    if view.payload is not None:
        for content in _iter_json_snippets(view.payload, max_items=400):
            score = _lexical_score(query_terms, content)
            if score <= 0:
                continue
            candidates.append(
                Slice(
                    doc_id=view.doc_id,
                    content=content,
                    score=score,
                    token_estimate=estimate_tokens(content),
                )
            )
    else:
        for para in _split_paragraphs(view.raw_text):
            content = para.strip()
            if not content:
                continue
            score = _lexical_score(query_terms, content)
            if score <= 0:
                continue
            candidates.append(
                Slice(
                    doc_id=view.doc_id,
                    content=content,
                    score=score,
                    token_estimate=estimate_tokens(content),
                )
            )

    candidates.sort(key=lambda s: (-s.score, s.token_estimate))
    return candidates[:top_k]


def _norm_terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t}


def _lexical_score(query_terms: set[str], content: str) -> float:
    terms = _norm_terms(content)
    if not terms:
        return 0.0
    overlap = len(query_terms & terms)
    if overlap == 0:
        return 0.0
    # Jaccard-like with small idf-ish bonus for short snippets
    jaccard = overlap / max(1, len(query_terms | terms))
    brevity_bonus = 1.0 / (1.0 + len(content) / 400.0)
    return jaccard + 0.25 * brevity_bonus


def _iter_json_snippets(payload: Any, max_items: int = 400) -> Iterable[str]:
    """Flatten JSON into concise 'path: value' snippets."""
    stack: list[tuple[tuple[str, ...], Any]] = [((), payload)]
    emitted = 0
    while stack and emitted < max_items:
        path, node = stack.pop()
        if isinstance(node, dict):
            for k, v in list(node.items())[:30]:
                stack.append((path + (str(k),), v))
        elif isinstance(node, list):
            for idx, v in enumerate(node[:30]):
                stack.append((path + (f"[{idx}]",), v))
        else:
            val = _format_scalar(node)
            if val:
                key = ".".join([p for p in path if not re.fullmatch(r"\[\d+\]", p)][-3:])
                snippet = f"{key}: {val}" if key else val
                emitted += 1
                yield snippet[:400]


def _format_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Collapse whitespace and trim
        s = re.sub(r"\s+", " ", s)
        return s[:300]
    return None


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _compress_slices(
    client: BedrockClient, view: DocumentView, slices: list[Slice], max_output_tokens: int
) -> CompressionResult:
    header = f"Document: {Path(view.path).name}\nQuestion: {view.question}\n"
    bullet_lines = [f"- {s.content}" for s in slices]
    prompt = header + "Snippets:\n" + "\n".join(bullet_lines[:20]) + "\n\n" + \
        "Produce a compact fact set capturing only information likely to answer the question."
    resp = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_COMPRESS,
        max_tokens=max_output_tokens,
        temperature=0.0,
    )
    content = resp.content.strip()
    return CompressionResult(summary=content, summary_tokens_estimate=estimate_tokens(content))


def _answer_with_compressed(
    client: BedrockClient, view: DocumentView, comp: CompressionResult, max_output_tokens: int
) -> QAResult:
    prompt = (
        f"Question: {view.question}\n"
        f"Facts:\n{comp.summary}\n\n"
        "Respond in JSON with keys 'answer' and 'confidence' ('high' or 'low')."
    )
    resp = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_ANSWER,
        max_tokens=max_output_tokens,
        temperature=0.0,
    )
    answer, conf = _parse_json_answer(resp.content)
    return QAResult(answer=answer, confidence=conf, stage="compressed")


def _answer_with_escalation(
    client: BedrockClient,
    view: DocumentView,
    comp: CompressionResult,
    slices: list[Slice],
    max_output_tokens: int,
) -> QAResult:
    header = f"Question: {view.question}\n"
    context = "Additional context:\n" + "\n".join([f"- {s.content}" for s in slices[:20]])
    prompt = (
        header
        + context
        + "\n\nFacts:\n"
        + comp.summary
        + "\n\nRespond in JSON with keys 'answer' and 'confidence' ('high' or 'low')."
    )
    resp = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_ANSWER,
        max_tokens=max_output_tokens,
        temperature=0.0,
    )
    answer, conf = _parse_json_answer(resp.content)
    return QAResult(answer=answer, confidence=conf, stage="escalated")


def _parse_json_answer(text: str) -> tuple[str, str]:
    # Try strict JSON first
    try:
        obj = json.loads(_extract_json_block(text))
        answer = str(obj.get("answer", "")).strip()
        confidence = str(obj.get("confidence", "")).strip().lower()
        if confidence not in {"high", "low"}:
            confidence = "low" if not answer else "high"
        return (answer or "unknown"), confidence
    except Exception:
        # Fallback: extract short answer and infer confidence by heuristic
        cleaned = text.strip()
        if not cleaned:
            return "unknown", "low"
        answer = cleaned.splitlines()[0][:200].strip()
        conf = "low" if any(x in cleaned.lower() for x in ["not sure", "unknown", "can't"]) else "high"
        return answer or "unknown", conf


def _extract_json_block(text: str) -> str:
    # Extract first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text


def _has_budget_headroom(usage: TokenUsage, budgets: Budgets, reserve: int) -> bool:
    total = usage.total_tokens or 0
    return total + reserve <= budgets.total_token_cap


if __name__ == "__main__":
    summary = run_iteration_budget_rag()
    print(json.dumps(summary.model_dump(), indent=2))


