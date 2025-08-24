from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .config import RunConfig, default_prompt
from .model_registry import DEFAULT_REGISTRY, ModelCard, list_models
from .latency_tester import (
	ModelLatencySummary,
	summarize_model,
)


logger = logging.getLogger("apispeedtest")


mcp = FastMCP("APISpeedTest MCP")


def _to_result_dict(r: ModelLatencySummary) -> Dict[str, Any]:
	return {
		"key": r.key,
		"provider": r.provider,
		"model": r.model,
		"nonstreaming_avg_s": r.nonstreaming_avg_s,
		"nonstreaming_runs": [vars(x) for x in r.nonstreaming_runs],
		"streaming_ttfb_avg_s": r.streaming_ttfb_avg_s,
		"streaming_total_avg_s": r.streaming_total_avg_s,
		"streaming_runs": [vars(x) for x in r.streaming_runs],
		"total_prompt_tokens": r.total_prompt_tokens,
		"total_completion_tokens": r.total_completion_tokens,
		"total_tokens": r.total_tokens,
		"nonstream_tokens_per_second": r.nonstream_tokens_per_second,
		"stream_tokens_per_second": r.stream_tokens_per_second,
	}


def _resolve_models(all_flag: bool, model_keys: Optional[List[str]]) -> List[str]:
	if all_flag:
		return list(list_models(DEFAULT_REGISTRY).keys())
	return model_keys or []


def _get_model_cards(requested: List[str]) -> Dict[str, ModelCard]:
	reg = list_models(DEFAULT_REGISTRY)
	missing: List[str] = [k for k in requested if k not in reg]
	if missing:
		raise ValueError(
			f"Unknown model keys: {', '.join(missing)}. Use list_models tool to see options."
		)
	return {k: reg[k] for k in requested}


@mcp.tool
def list_models_tool() -> Dict[str, Any]:
	"""List available model keys and metadata."""
	reg = list_models(DEFAULT_REGISTRY)
	data: List[Dict[str, Any]] = []
	for key, card in reg.items():
		data.append({
			"key": key,
			"provider": card.provider,
			"model": card.model,
			"supports_streaming": card.supports_streaming,
			"default_kwargs": card.default_kwargs,
			"description": card.description,
		})
	return {"models": data}


@mcp.tool
def run_benchmark(
	prompt: Optional[str] = None,
	prompt_file: Optional[str] = None,
	models: Optional[List[str]] = None,
	all_models: bool = False,
	runs: int = 3,
	mode: str = "both",
	request_timeout_seconds: Optional[float] = None,
	model_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
	"""Run latency benchmarks for selected models and return structured results.

	Arguments:
	- prompt: Prompt text. If omitted, uses default. If prompt_file is set, that takes precedence.
	- prompt_file: Path to a file containing the prompt text.
	- models: Specific model keys to run (e.g., ["openai:gpt-4o-mini"]).
	- all_models: If true, runs all default models and ignores `models`.
	- runs: Number of runs per model.
	- mode: "both" | "nonstream" | "stream".
	- request_timeout_seconds: Optional per-request timeout.
	- model_overrides: Per-model overrides mapping by model key.
	"""
	if prompt_file:
		with open(prompt_file, "r", encoding="utf-8") as f:
			prompt_text = f.read()
	else:
		prompt_text = prompt or default_prompt()

	selected_models = _resolve_models(all_models, models)
	cfg = RunConfig(
		prompt=prompt_text,
		models=selected_models,
		runs=runs,
		mode=mode,
		request_timeout_seconds=request_timeout_seconds,
		model_overrides=model_overrides or {},
	)
	cfg.validate()

	model_cards = _get_model_cards(cfg.models)
	results: List[ModelLatencySummary] = []
	for key, card in model_cards.items():
		try:
			res = summarize_model(
				model_key=key,
				card=card,
				prompt=cfg.prompt,
				runs=cfg.runs,
				mode=cfg.mode,
				request_timeout_seconds=cfg.request_timeout_seconds,
				model_overrides=cfg.model_overrides.get(key),
			)
			exception: Exception | None = None
		except Exception as e:
			logger.error("Model %s failed: %s", key, str(e))
			exception = e
			res = None
		if res is not None:
			results.append(res)

	return {
		"results": [_to_result_dict(r) for r in results],
	}


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	mcp.run()


if __name__ == "__main__":
	main()

