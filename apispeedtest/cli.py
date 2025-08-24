from __future__ import annotations

import argparse
import json
import os
import logging
from typing import Any, Dict, List, Optional

from .config import RunConfig, default_prompt
from .model_registry import DEFAULT_REGISTRY, ModelCard, list_models
from .latency_tester import (
	ModelLatencySummary,
	print_human_readable,
	summarize_model,
	write_csv,
	write_json,
)


def _load_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
	if prompt_file:
		with open(prompt_file, "r", encoding="utf-8") as f:
			return f.read()
	return prompt or default_prompt()


def _resolve_models(all_flag: bool, model_keys: List[str]) -> List[str]:
	if all_flag:
		return list(list_models(DEFAULT_REGISTRY).keys())
	return model_keys


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Benchmark LLM API latency via LangChain (streaming and non-streaming)",
	)
	parser.add_argument("command", nargs="?", default="run", choices=["run", "list-models"], help="Command to run")
	parser.add_argument("--config", dest="config_path", help="Path to YAML config file")
	parser.add_argument("-m", "--models", dest="models", help="Comma-separated model keys, e.g., openai:gpt-4o-mini,anthropic:claude-3-5-sonnet-latest")
	parser.add_argument("--all", dest="all_models", action="store_true", help="Run all default models")
	parser.add_argument("--prompt", dest="prompt", help="Prompt text")
	parser.add_argument("--prompt-file", dest="prompt_file", help="Path to prompt file")
	parser.add_argument("--runs", dest="runs", type=int, default=3, help="Number of runs per model")
	parser.add_argument("--mode", dest="mode", choices=["both", "nonstream", "stream"], default="both", help="Which latency tests to run")
	parser.add_argument("--request-timeout", dest="request_timeout", type=float, help="Request timeout (seconds)")
	parser.add_argument("--json-out", dest="json_out", help="Write results to JSON file")
	parser.add_argument("--csv-out", dest="csv_out", help="Write results to CSV file")
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Enable verbose (DEBUG) logging")
	return parser.parse_args()


def _get_model_cards(requested: List[str]) -> Dict[str, ModelCard]:
	reg = list_models(DEFAULT_REGISTRY)
	missing: List[str] = [k for k in requested if k not in reg]
	if missing:
		raise SystemExit(f"Unknown model keys: {', '.join(missing)}. Run 'apispeedtest list-models' to see options.")
	return {k: reg[k] for k in requested}


def _load_config_from_args(args: argparse.Namespace) -> RunConfig:
	if args.config_path:
		cfg = RunConfig.from_yaml(args.config_path)
		return cfg

	models = []
	if args.models:
		models = [m.strip() for m in args.models.split(",") if m.strip()]
	models = _resolve_models(args.all_models, models)

	prompt = _load_prompt(args.prompt, args.prompt_file)
	return RunConfig(
		prompt=prompt,
		models=models,
		runs=args.runs,
		mode=args.mode,
		request_timeout_seconds=args.request_timeout,
		model_overrides={},
	)


def _print_model_list() -> None:
	reg = list_models(DEFAULT_REGISTRY)
	print("Available models:")
	for key, card in reg.items():
		desc = f" - {card.description}" if card.description else ""
		print(f"- {key}{desc}")


def _setup_logging(verbose: bool) -> None:
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="%(asctime)s %(levelname)s %(message)s",
	)


def main() -> None:
	args = _parse_args()
	if args.command == "list-models":
		_print_model_list()
		return

	_setup_logging(args.verbose)
	logger = logging.getLogger("apispeedtest")

	cfg = _load_config_from_args(args)
	cfg.validate()
	logger.info("Starting benchmark: models=%s, runs=%d, mode=%s, timeout=%s", 
		", ".join(cfg.models) if cfg.models else "(none)", cfg.runs, cfg.mode, str(cfg.request_timeout_seconds))

	model_cards = _get_model_cards(cfg.models)
	results: List[ModelLatencySummary] = []
	model_total = len(model_cards)
	for idx, (key, card) in enumerate(model_cards.items(), start=1):
		overrides = cfg.model_overrides.get(key)
		logger.info("[%d/%d] Running model %s (%s - %s)", idx, model_total, key, card.provider, card.model)
		try:
			res = summarize_model(
				model_key=key,
				card=card,
				prompt=cfg.prompt,
				runs=cfg.runs,
				mode=cfg.mode,
				request_timeout_seconds=cfg.request_timeout_seconds,
				model_overrides=overrides,
			)
			exception: Exception | None = None
		except Exception as e:
			logger.error("Model %s failed: %s", key, str(e))
			exception = e
			res = None
		if res is not None:
			results.append(res)
			logger.info("Completed model %s", key)
		else:
			logger.warning("Skipping model %s due to error", key)

	print_human_readable(results)

	if args.json_out:
		write_json(args.json_out, results)
		logger.info("Wrote JSON results to %s", args.json_out)
	if args.csv_out:
		write_csv(args.csv_out, results)
		logger.info("Wrote CSV results to %s", args.csv_out)


if __name__ == "__main__":
	main()

