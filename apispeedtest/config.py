from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RunConfig:
	prompt: str
	models: List[str]
	runs: int = 3
	mode: str = "both"  # both | nonstream | stream
	request_timeout_seconds: Optional[float] = None
	model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

	@staticmethod
	def from_yaml(path: str) -> "RunConfig":
		with open(path, "r", encoding="utf-8") as f:
			data = yaml.safe_load(f) or {}
		return RunConfig(
			prompt=data.get("prompt", default_prompt()),
			models=list(data.get("models", [])),
			runs=int(data.get("runs", 3)),
			mode=str(data.get("mode", "both")),
			request_timeout_seconds=data.get("request_timeout_seconds"),
			model_overrides=dict(data.get("model_overrides", {})),
		)

	def validate(self) -> None:
		if self.mode not in {"both", "nonstream", "stream"}:
			raise ValueError("mode must be one of: both | nonstream | stream")
		if not self.models:
			raise ValueError("No models specified. Use --all or provide --models / config file.")
		if self.runs < 1:
			raise ValueError("runs must be >= 1")


def default_prompt() -> str:
	return (
		"You are a helpful assistant. In 3 concise bullet points, explain how to boil pasta "
		"perfectly every time."
	)

