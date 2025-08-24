from __future__ import annotations

import csv
import json
import statistics
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from .model_registry import ModelCard, create_chat_model


logger = logging.getLogger("apispeedtest")


@dataclass
class NonStreamingRun:
	duration_seconds: float
	prompt_tokens: int | None = None
	completion_tokens: int | None = None
	total_tokens: int | None = None


@dataclass
class StreamingRun:
	time_to_first_token_seconds: float
	total_time_seconds: float
	num_chunks: int
	characters_streamed: int
	prompt_tokens: int | None = None
	completion_tokens: int | None = None
	total_tokens: int | None = None


@dataclass
class ModelLatencySummary:
	key: str
	provider: str
	model: str
	nonstreaming_avg_s: Optional[float]
	nonstreaming_runs: List[NonStreamingRun]
	streaming_ttfb_avg_s: Optional[float]
	streaming_total_avg_s: Optional[float]
	streaming_runs: List[StreamingRun]
	# Aggregated token stats
	total_prompt_tokens: Optional[int]
	total_completion_tokens: Optional[int]
	total_tokens: Optional[int]
	nonstream_tokens_per_second: Optional[float]
	stream_tokens_per_second: Optional[float]


def _now() -> float:
	return time.perf_counter()


def _sum_defined(ints: List[Optional[int]]) -> Optional[int]:
	vals = [v for v in ints if isinstance(v, int)]
	return sum(vals) if vals else None


def _estimate_tokens_from_text(text: str) -> int:
	# Heuristic fallback when provider usage is unavailable
	return max(1, int(round(len(text) / 4)))


def measure_non_streaming(llm: BaseChatModel, prompt: str) -> NonStreamingRun:
	start = _now()
	resp = llm.invoke(prompt)
	end = _now()

	usage = getattr(resp, "usage_metadata", None) or {}
	prompt_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
	completion_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
	total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None

	# Fallback estimation if usage not provided
	if prompt_tokens is None:
		prompt_tokens = _estimate_tokens_from_text(prompt)
	if completion_tokens is None:
		content = getattr(resp, "content", "")
		if isinstance(content, str):
			completion_tokens = _estimate_tokens_from_text(content)
		else:
			completion_tokens = None
	if total_tokens is None and (prompt_tokens is not None and completion_tokens is not None):
		total_tokens = prompt_tokens + completion_tokens

	return NonStreamingRun(
		duration_seconds=end - start,
		prompt_tokens=prompt_tokens,
		completion_tokens=completion_tokens,
		total_tokens=total_tokens,
	)


def measure_streaming(llm: BaseChatModel, prompt: str) -> StreamingRun:
	start = _now()
	ttfb: Optional[float] = None
	num_chunks = 0
	chars = 0
	full_text_parts: List[str] = []
	usage_last: Dict[str, Any] | None = None
	for chunk in llm.stream(prompt):
		if ttfb is None:
			ttfb = _now() - start
		# chunk can be AIMessageChunk or similar; content may be str or list[BaseMessage]
		content = getattr(chunk, "content", "")
		if isinstance(content, str):
			chars += len(content)
			full_text_parts.append(content)
		elif isinstance(content, list):
			# Some providers return content parts
			for part in content:
				text = getattr(part, "text", "")
				chars += len(text or "")
				if text:
					full_text_parts.append(text)
		# Try to capture usage metadata if present on chunk
		chunk_usage = getattr(chunk, "usage_metadata", None)
		if isinstance(chunk_usage, dict) and chunk_usage:
			usage_last = chunk_usage
		num_chunks += 1
	end = _now()
	if ttfb is None:
		# No chunks produced; treat as zero-length first token
		ttfb = end - start

	# Prompt tokens: prefer usage; else estimate from prompt text
	prompt_tokens: int | None = None
	completion_tokens: int | None = None
	total_tokens: int | None = None
	if usage_last:
		prompt_tokens = usage_last.get("input_tokens")
		completion_tokens = usage_last.get("output_tokens")
		total_tokens = usage_last.get("total_tokens")
	if prompt_tokens is None:
		prompt_tokens = _estimate_tokens_from_text(prompt)
	if completion_tokens is None:
		completion_text = "".join(full_text_parts)
		completion_tokens = _estimate_tokens_from_text(completion_text) if completion_text else None
	if total_tokens is None and (prompt_tokens is not None and completion_tokens is not None):
		total_tokens = prompt_tokens + completion_tokens

	return StreamingRun(
		time_to_first_token_seconds=ttfb,
		total_time_seconds=end - start,
		num_chunks=num_chunks,
		characters_streamed=chars,
		prompt_tokens=prompt_tokens,
		completion_tokens=completion_tokens,
		total_tokens=total_tokens,
	)


def summarize_model(
	model_key: str,
	card: ModelCard,
	prompt: str,
	runs: int,
	mode: str,
	request_timeout_seconds: Optional[float] = None,
	model_overrides: Optional[Dict[str, Any]] = None,
) -> ModelLatencySummary:
	logger.debug("Creating chat model for %s (%s - %s)", model_key, card.provider, card.model)
	llm = create_chat_model(
		card=card,
		streaming=True,  # we can use stream() regardless; invoke() will ignore
		request_timeout_seconds=request_timeout_seconds,
		model_overrides=model_overrides,
	)

	nonstream_runs: List[NonStreamingRun] = []
	stream_runs: List[StreamingRun] = []

	for i in range(1, runs + 1):
		if mode in {"both", "nonstream"}:
			logger.info("%s non-stream run %d/%d: start", model_key, i, runs)
			ns = measure_non_streaming(llm, prompt)
			nonstream_runs.append(ns)
			logger.info("%s non-stream run %d/%d: %.3fs (tokens: prompt=%s, completion=%s, total=%s)",
				model_key, i, runs, ns.duration_seconds, str(ns.prompt_tokens), str(ns.completion_tokens), str(ns.total_tokens))
		if mode in {"both", "stream"}:
			logger.info("%s stream run %d/%d: start", model_key, i, runs)
			st = measure_streaming(llm, prompt)
			stream_runs.append(st)
			logger.info(
				"%s stream run %d/%d: ttfb=%.3fs total=%.3fs chunks=%d (tokens: prompt=%s, completion=%s, total=%s)",
				model_key, i, runs, st.time_to_first_token_seconds, st.total_time_seconds, st.num_chunks,
				str(st.prompt_tokens), str(st.completion_tokens), str(st.total_tokens),
			)

	non_avg = (
		statistics.fmean(r.duration_seconds for r in nonstream_runs)
		if nonstream_runs
		else None
	)
	stream_ttfb_avg = (
		statistics.fmean(r.time_to_first_token_seconds for r in stream_runs)
		if stream_runs
		else None
	)
	stream_total_avg = (
		statistics.fmean(r.total_time_seconds for r in stream_runs)
		if stream_runs
		else None
	)

	# Totals across whichever modes ran
	total_prompt_tokens = _sum_defined(
		[r.prompt_tokens for r in nonstream_runs] + [r.prompt_tokens for r in stream_runs]
	)
	total_completion_tokens = _sum_defined(
		[r.completion_tokens for r in nonstream_runs] + [r.completion_tokens for r in stream_runs]
	)
	total_tokens_all = _sum_defined(
		[r.total_tokens for r in nonstream_runs] + [r.total_tokens for r in stream_runs]
	)

	# Throughput (tokens per second) using completion tokens only
	ns_tokens = [r.completion_tokens for r in nonstream_runs]
	ns_durations = [r.duration_seconds for r in nonstream_runs]
	if any(isinstance(x, int) for x in ns_tokens):
		tokens_sum = sum(int(x) for x in ns_tokens if isinstance(x, int))
		dur_sum = sum(ns_durations)
		nonstream_tps = (tokens_sum / dur_sum) if dur_sum > 0 else None
	else:
		nonstream_tps = None

	st_tokens = [r.completion_tokens for r in stream_runs]
	st_durations = [r.total_time_seconds for r in stream_runs]
	if any(isinstance(x, int) for x in st_tokens):
		tokens_sum = sum(int(x) for x in st_tokens if isinstance(x, int))
		dur_sum = sum(st_durations)
		stream_tps = (tokens_sum / dur_sum) if dur_sum > 0 else None
	else:
		# Fallback: approximate from characters if tokens unknown
		chars_sum = sum(r.characters_streamed for r in stream_runs)
		dur_sum = sum(st_durations)
		stream_tps = ((chars_sum / 4) / dur_sum) if dur_sum > 0 and chars_sum > 0 else None

	return ModelLatencySummary(
		key=model_key,
		provider=card.provider,
		model=card.model,
		nonstreaming_avg_s=non_avg,
		nonstreaming_runs=nonstream_runs,
		streaming_ttfb_avg_s=stream_ttfb_avg,
		streaming_total_avg_s=stream_total_avg,
		streaming_runs=stream_runs,
		total_prompt_tokens=total_prompt_tokens,
		total_completion_tokens=total_completion_tokens,
		total_tokens=total_tokens_all,
		nonstream_tokens_per_second=nonstream_tps,
		stream_tokens_per_second=stream_tps,
	)



def print_human_readable(results: List[ModelLatencySummary]) -> None:
	def _format_table(headers: List[str], rows: List[List[str]], right_align_cols: set[int]) -> str:
		# Compute column widths based on content and headers
		widths: List[int] = []
		for i, h in enumerate(headers):
			max_cell = max((len(r[i]) for r in rows), default=0)
			widths.append(max(len(h), max_cell))

		# Box drawing characters
		h = "─"
		v = "│"
		top_left, top_mid, top_right = "┌", "┬", "┐"
		mid_left, mid_mid, mid_right = "├", "┼", "┤"
		bot_left, bot_mid, bot_right = "└", "┴", "┘"

		def _border(left: str, mid: str, right: str) -> str:
			parts = [(h * (w + 2)) for w in widths]
			return left + mid.join(parts) + right

		def _row(cells: List[str], is_header: bool = False) -> str:
			formatted: List[str] = []
			for i, cell in enumerate(cells):
				if i in right_align_cols and not is_header:
					formatted.append(f" {cell.rjust(widths[i])} ")
				else:
					formatted.append(f" {cell.ljust(widths[i])} ")
			return v + v.join(formatted) + v

		lines: List[str] = []
		lines.append(_border(top_left, top_mid, top_right))
		lines.append(_row(headers, is_header=True))
		lines.append(_border(mid_left, mid_mid, mid_right))
		for r in rows:
			lines.append(_row(r))
		lines.append(_border(bot_left, bot_mid, bot_right))
		return "\n".join(lines)

	headers = [
		"Model Key",
		"Provider",
		"Model",
		"NS avg (s)",
		"NS TPS",
		"TTFB (s)",
		"Stream (s)",
		"ST TPS",
	]
	rows: List[List[str]] = []
	for r in results:
		non_avg = f"{r.nonstreaming_avg_s:.3f}" if r.nonstreaming_avg_s is not None else "-"
		ns_tps = f"{r.nonstream_tokens_per_second:.1f}" if r.nonstream_tokens_per_second is not None else "-"
		ttfb_avg = f"{r.streaming_ttfb_avg_s:.3f}" if r.streaming_ttfb_avg_s is not None else "-"
		total_avg = f"{r.streaming_total_avg_s:.3f}" if r.streaming_total_avg_s is not None else "-"
		st_tps = f"{r.stream_tokens_per_second:.1f}" if r.stream_tokens_per_second is not None else "-"
		rows.append([
			r.key,
			r.provider,
			r.model,
			non_avg,
			ns_tps,
			ttfb_avg,
			total_avg,
			st_tps,
		])

	# Right-align numeric columns
	right_align = {3, 4, 5, 6, 7}
	print("\nLatency Results (seconds):")
	print(_format_table(headers, rows, right_align))

	print("\nLegend:")
	print("  - NS avg (s): Average end-to-end time for a non-streaming request.")
	print("  - NS TPS: Non-streaming throughput in tokens/sec (completion tokens / time).")
	print("  - TTFB (s): Time-to-first-token for streaming.")
	print("  - Stream (s): Total streaming time.")
	print("  - ST TPS: Streaming throughput in tokens/sec (completion tokens / time; estimated if usage missing).")
	# Totals
	total_prompt = _sum_defined([r.total_prompt_tokens for r in results])
	total_completion = _sum_defined([r.total_completion_tokens for r in results])
	print("\nTotals across selected models and runs:")
	print(f"  - Prompt tokens: {total_prompt if total_prompt is not None else '-'}")
	print(f"  - Completion tokens: {total_completion if total_completion is not None else '-'}")


def write_json(path: str, results: List[ModelLatencySummary]) -> None:
	def to_dict(r: ModelLatencySummary) -> Dict[str, Any]:
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

	with open(path, "w", encoding="utf-8") as f:
		json.dump([to_dict(r) for r in results], f, indent=2)


def write_csv(path: str, results: List[ModelLatencySummary]) -> None:
	rows: List[Dict[str, Any]] = []
	for r in results:
		row = {
			"key": r.key,
			"provider": r.provider,
			"model": r.model,
			"nonstreaming_avg_s": r.nonstreaming_avg_s,
			"streaming_ttfb_avg_s": r.streaming_ttfb_avg_s,
			"streaming_total_avg_s": r.streaming_total_avg_s,
		}
		rows.append(row)

	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"key",
				"provider",
				"model",
				"nonstreaming_avg_s",
				"nonstream_tokens_per_second",
				"streaming_ttfb_avg_s",
				"streaming_total_avg_s",
				"stream_tokens_per_second",
				"total_prompt_tokens",
				"total_completion_tokens",
				"total_tokens",
			],
		)
		writer.writeheader()
		# enrich rows
		enriched: List[Dict[str, Any]] = []
		for r in results:
			enriched.append({
				"key": r.key,
				"provider": r.provider,
				"model": r.model,
				"nonstreaming_avg_s": r.nonstreaming_avg_s,
				"nonstream_tokens_per_second": r.nonstream_tokens_per_second,
				"streaming_ttfb_avg_s": r.streaming_ttfb_avg_s,
				"streaming_total_avg_s": r.streaming_total_avg_s,
				"stream_tokens_per_second": r.stream_tokens_per_second,
				"total_prompt_tokens": r.total_prompt_tokens,
				"total_completion_tokens": r.total_completion_tokens,
				"total_tokens": r.total_tokens,
			})
		writer.writerows(enriched)

