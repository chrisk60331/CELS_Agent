"""Agent run summary for tracking token usage and metadata."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentRunSummary(BaseModel):
    """Summary of an agent run with token usage and metadata."""

    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    usage: Optional[Dict[str, Any]] = Field(None, description="Detailed usage metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @classmethod
    def from_usage(
        cls,
        usage: Optional[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentRunSummary":
        """Create summary from usage dictionary."""
        normalized_usage, total_tokens = cls._normalize_usage_payload(usage)

        return cls(
            total_tokens=total_tokens,
            usage=normalized_usage,
            metadata=metadata or {},
        )

    @staticmethod
    def _normalize_usage_payload(
        usage: Optional[Any],
    ) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
        """Coerce agent usage payloads into a consistent dictionary."""
        if usage is None:
            return None, None

        usage_dict: Optional[Dict[str, Any]] = None
        if isinstance(usage, dict):
            usage_dict = dict(usage)
        else:
            total_tokens_attr = getattr(usage, "total_tokens", None)
            if total_tokens_attr is not None:
                usage_dict = {
                    "total_tokens": total_tokens_attr,
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                }
            elif hasattr(usage, "dict"):
                try:
                    usage_dict = dict(usage.dict())
                except TypeError:
                    usage_dict = None
            elif hasattr(usage, "__dict__"):
                usage_dict = dict(vars(usage))

        if not usage_dict:
            return None, None

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        # Mirror camelCase keys with snake_case equivalents when present.
        key_pairs = [
            ("inputTokens", "input_tokens"),
            ("outputTokens", "output_tokens"),
            ("totalTokens", "total_tokens"),
        ]
        for source_key, target_key in key_pairs:
            if source_key in usage_dict and target_key not in usage_dict:
                usage_dict[target_key] = usage_dict[source_key]

        # Coerce numeric token counts.
        for key in {"input_tokens", "output_tokens", "total_tokens"}:
            if key in usage_dict:
                coerced_value = _coerce_int(usage_dict.get(key))
                if coerced_value is not None:
                    usage_dict[key] = coerced_value

        total_tokens = usage_dict.get("total_tokens")
        if total_tokens is None:
            input_tokens = usage_dict.get("input_tokens")
            output_tokens = usage_dict.get("output_tokens")
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
                usage_dict["total_tokens"] = total_tokens

        return usage_dict, _coerce_int(total_tokens)

