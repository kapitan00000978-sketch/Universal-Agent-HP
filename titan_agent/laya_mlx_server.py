"""Compatibility adapter for the optional local Laya-MLX decision runtime."""
from __future__ import annotations

from typing import Any

from .laya_decisions import predict_decisions


class LayaMLXConnector:
    """Expose genuine Laya-MLX typed inference; never fabricate chat responses."""

    def __init__(self, model_path: str = "aac6fef/laya-mlx"):
        self.model_path = model_path

    async def predict(self, state: str, questions: dict[str, Any]) -> str:
        """Run a local typed-decision request without an API key."""
        return await predict_decisions(
            state,
            questions,
            backend="laya-mlx",
            model=self.model_path,
        )

    async def generate_response(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Reject the old chat-shaped API: Laya does not generate free-form text."""
        raise RuntimeError(
            "Laya-MLX produces typed choice/score/yes-no decisions, not chat text. "
            "Use predict(state, questions) or the laya_decide tool."
        )
