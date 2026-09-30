"""Optional, API-key-free typed-decision adapters for upstream Laya runtimes.

Laya produces structured choices, scores and yes/no probabilities. It is not a
free-form text-generation model and must not be presented as a chat completion
backend. The standard runtime works where its PyTorch dependencies are
available; the MLX runtime is limited to supported Apple Silicon Macs.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import platform
import threading
from collections.abc import Mapping
from typing import Any

_LOAD_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.RLock()
_RUNTIMES: dict[tuple[str, str], Any] = {}


def _default_backend() -> str:
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        if importlib.util.find_spec("laya_mlx") is not None:
            return "laya-mlx"
    return "laya"


def _load_runtime(backend: str, model: str) -> Any:
    key = (backend, model)
    with _LOAD_LOCK:
        if key in _RUNTIMES:
            return _RUNTIMES[key]
        if backend == "laya-mlx":
            if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
                raise RuntimeError("laya-mlx requires a supported Apple Silicon Mac running macOS 14 or newer.")
            try:
                import laya_mlx
            except ImportError as exc:
                raise RuntimeError("Install the optional Apple Silicon backend with `pip install .[laya-mlx]`.") from exc
            runtime = laya_mlx.load(model or "aac6fef/laya-mlx")
        elif backend == "laya":
            try:
                from laya import Router
            except ImportError as exc:
                raise RuntimeError("Install the optional Laya backend with `pip install .[laya]`.") from exc
            runtime = Router()
        else:
            raise ValueError("backend must be 'auto', 'laya', or 'laya-mlx'.")
        _RUNTIMES[key] = runtime
        return runtime


def _predict_sync(state: str, questions: dict[str, Any], backend: str, model: str) -> dict[str, Any]:
    selected = _default_backend() if backend == "auto" else backend
    runtime = _load_runtime(selected, model)
    # Serialize calls because the upstream model runtimes may retain mutable
    # batching/router state. This favors correctness over parallel throughput.
    with _INFERENCE_LOCK:
        if selected == "laya" and model:
            result = runtime.predict(state, questions, model=model)
        else:
            result = runtime.predict(state, questions)
    if isinstance(result, Mapping):
        return {"backend": selected, "result": dict(result)}
    return {"backend": selected, "result": result}


async def predict_decisions(
    state: str,
    questions: dict[str, Any],
    backend: str = "auto",
    model: str = "",
) -> str:
    """Run local typed decisions asynchronously; no API key is read or sent."""
    if not isinstance(state, str) or not state.strip():
        return "Error: state must be a non-empty string."
    if len(state) > 100_000:
        return "Error: state exceeds the 100000-character limit."
    if not isinstance(questions, dict) or not questions:
        return "Error: questions must be a non-empty object of typed Laya questions."
    if len(questions) > 20:
        return "Error: at most 20 questions can be sent in one call."
    try:
        serialized_questions = json.dumps(questions, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        return f"Error: questions must be JSON-serializable: {exc!s}"
    if len(serialized_questions) > 50_000:
        return "Error: question schema exceeds the 50000-character limit."
    if backend not in {"auto", "laya", "laya-mlx"}:
        return "Error: backend must be one of: auto, laya, laya-mlx."
    if not isinstance(model, str) or len(model) > 200:
        return "Error: model must be a string of at most 200 characters."
    selected = _default_backend() if backend == "auto" else backend
    try:
        payload = await asyncio.to_thread(_predict_sync, state, questions, selected, model)
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception as primary_exc:  # noqa: BLE001 - optional backend errors become clear tool results
        if backend == "auto" and selected == "laya-mlx":
            try:
                payload = await asyncio.to_thread(_predict_sync, state, questions, "laya", model)
                payload["fallback_from"] = "laya-mlx"
                payload["fallback_reason"] = f"{type(primary_exc).__name__}: {primary_exc!s}"[:500]
                return json.dumps(payload, ensure_ascii=False, default=str)
            except Exception as fallback_exc:  # noqa: BLE001
                return (
                    "Error: both local Laya backends failed. "
                    f"MLX: {type(primary_exc).__name__}: {primary_exc!s}; "
                    f"Laya: {type(fallback_exc).__name__}: {fallback_exc!s}"
                )
        return f"Error: local Laya decision inference failed: {type(primary_exc).__name__}: {primary_exc!s}"


def main(argv: list[str] | None = None) -> int:
    """Small keyless CLI for structured decisions when no chat API is configured."""
    parser = argparse.ArgumentParser(
        description="Run local Laya typed decisions without an API key (not a chat/code generator)."
    )
    parser.add_argument("--state", required=True, help="Text/document to evaluate")
    parser.add_argument("--questions", required=True, help="Path to a JSON object containing Laya questions")
    parser.add_argument("--backend", choices=["auto", "laya", "laya-mlx"], default="auto")
    parser.add_argument("--model", default="", help="Optional checkpoint ID (primarily for laya-mlx)")
    args = parser.parse_args(argv)
    try:
        with open(args.questions, encoding="utf-8") as handle:
            questions = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot load questions JSON: {exc}")
    if not isinstance(questions, dict):
        parser.error("questions JSON must be an object")
    output = asyncio.run(predict_decisions(args.state, questions, args.backend, args.model))
    print(output)
    return 1 if output.startswith("Error:") else 0
