from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
import logging


@dataclass
class ModelCard:
	key: str
	provider: str  # openai | azure | anthropic | gemini | llama (groq)
	model: str
	default_kwargs: Dict[str, Any] = field(default_factory=dict)
	supports_streaming: bool = True
	description: Optional[str] = None


def _build_openai(model: str, streaming: bool, request_timeout: Optional[float], **kwargs: Any):
	from langchain_openai import ChatOpenAI

	init_kwargs: Dict[str, Any] = {"model": model, **kwargs}
	if request_timeout is not None:
		init_kwargs["timeout"] = request_timeout
	# Ensure we request usage metadata where supported
	init_kwargs.setdefault("default_headers", {})  # harmless for providers
	# Some OpenAI models only accept the default temperature (1). If a non-default
	# temperature is provided, drop it to avoid 400 errors.
	temp = init_kwargs.get("temperature")
	if isinstance(temp, (int, float)) and temp != 1:
		logging.getLogger("apispeedtest").debug(
			"Dropping unsupported 'temperature'=%s for OpenAI model %s", str(temp), model
		)
		init_kwargs.pop("temperature", None)
	return ChatOpenAI(**init_kwargs)


def _build_azure_openai(model: str, streaming: bool, request_timeout: Optional[float], **kwargs: Any):
	from langchain_openai import ChatOpenAI

	# For Azure, "model" is typically the deployment name
	init_kwargs: Dict[str, Any] = {"model": model, **kwargs}
	if request_timeout is not None:
		init_kwargs["timeout"] = request_timeout
	init_kwargs.setdefault("default_headers", {})

	# Some Azure deployments mirror OpenAI temperature constraints
	temp = init_kwargs.get("temperature")
	if isinstance(temp, (int, float)) and temp != 1:
		logging.getLogger("apispeedtest").debug(
			"Dropping unsupported 'temperature'=%s for Azure OpenAI deployment %s", str(temp), model
		)
		init_kwargs.pop("temperature", None)

	# Expect azure configuration via env or kwargs:
	#  - AZURE_OPENAI_API_KEY / api_key
	#  - AZURE_OPENAI_ENDPOINT or azure_endpoint
	#  - OPENAI_API_VERSION or openai_api_version
	return ChatOpenAI(**init_kwargs)


def _build_anthropic(model: str, streaming: bool, request_timeout: Optional[float], **kwargs: Any):
	from langchain_anthropic import ChatAnthropic

	init_kwargs: Dict[str, Any] = {"model": model, **kwargs}
	if request_timeout is not None:
		init_kwargs["timeout"] = request_timeout
	return ChatAnthropic(**init_kwargs)


def _build_gemini(model: str, streaming: bool, request_timeout: Optional[float], **kwargs: Any):
	from langchain_google_genai import ChatGoogleGenerativeAI

	init_kwargs: Dict[str, Any] = {"model": model, **kwargs}
	if request_timeout is not None:
		init_kwargs["timeout"] = request_timeout
	return ChatGoogleGenerativeAI(**init_kwargs)


def _build_groq(model: str, streaming: bool, request_timeout: Optional[float], **kwargs: Any):
	from langchain_groq import ChatGroq

	init_kwargs: Dict[str, Any] = {"model": model, **kwargs}
	if request_timeout is not None:
		init_kwargs["timeout"] = request_timeout
	return ChatGroq(**init_kwargs)


PROVIDER_BUILDERS: Mapping[str, Any] = {
	"openai": _build_openai,
	"azure": _build_azure_openai,
	"azure-openai": _build_azure_openai,
	"anthropic": _build_anthropic,
	"gemini": _build_gemini,
	"llama": _build_groq,  # llama via Groq
}


DEFAULT_REGISTRY: Dict[str, ModelCard] = {
	"openai:gpt-4o-mini": ModelCard(
		key="openai:gpt-4o-mini",
		provider="openai",
		model="gpt-4o-mini",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-4o mini",
	),
	"openai:gpt-4o": ModelCard(
		key="openai:gpt-4o",
		provider="openai",
		model="gpt-4o",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-4o",
	),
	"openai:gpt-4.1": ModelCard(
		key="openai:gpt-4.1",
		provider="openai",
		model="gpt-4.1",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-4.1",
	),
	"openai:gpt-3.5-turbo": ModelCard(
		key="openai:gpt-3.5-turbo",
		provider="openai",
		model="gpt-3.5-turbo",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-3.5 Turbo",
	),
	"openai:gpt-3.5-turbo-16k": ModelCard(
		key="openai:gpt-3.5-turbo-16k",
		provider="openai",
		model="gpt-3.5-turbo-16k",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-3.5 Turbo 16K (legacy)",
	),
	"openai:gpt-4.1-mini": ModelCard(
		key="openai:gpt-4.1-mini",
		provider="openai",
		model="gpt-4.1-mini",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-4.1 Mini",
	),
	"openai:gpt-4-turbo": ModelCard(
		key="openai:gpt-4-turbo",
		provider="openai",
		model="gpt-4-turbo",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-4 Turbo",
	),
	"openai:gpt-4": ModelCard(
		key="openai:gpt-4",
		provider="openai",
		model="gpt-4",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-4 (legacy)",
	),
	"openai:o3": ModelCard(
		key="openai:o3",
		provider="openai",
		model="o3",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI O3 (reasoning)",
	),
	"openai:o3-mini": ModelCard(
		key="openai:o3-mini",
		provider="openai",
		model="o3-mini",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI O3 Mini (reasoning)",
	),
	"openai:o4-mini": ModelCard(
		key="openai:o4-mini",
		provider="openai",
		model="o4-mini",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI O4 Mini",
	),
	"openai:gpt-4o-2024-08-06": ModelCard(
		key="openai:gpt-4o-2024-08-06",
		provider="openai",
		model="gpt-4o-2024-08-06",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-4o (2024-08-06)",
	),
	"openai:gpt-5": ModelCard(
		key="openai:gpt-5",
		provider="openai",
		model="gpt-5",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="OpenAI GPT-5",
	),
	# Azure OpenAI (deployment names). Configure via env vars or overrides:
	#   AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION
	#   Or pass azure_endpoint/openai_api_version/api_key in model_overrides.
	"azure:gpt-4o-mini": ModelCard(
		key="azure:gpt-4o-mini",
		provider="azure",
		model="gpt-4o-mini",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="Azure OpenAI deployment: gpt-4o-mini",
	),
	"azure:gpt-4o": ModelCard(
		key="azure:gpt-4o",
		provider="azure",
		model="gpt-4o",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="Azure OpenAI deployment: gpt-4o",
	),
	"anthropic:claude-3-7-sonnet-latest": ModelCard(
		key="anthropic:claude-3-5-sonnet-latest",
		provider="anthropic",
		model="claude-3-7-sonnet-latest",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="Anthropic Claude 3.5 Sonnet",
	),
	"gemini:gemini-1.5-pro": ModelCard(
		key="gemini:gemini-1.5-pro",
		provider="gemini",
		model="gemini-1.5-pro",
		default_kwargs={"temperature": 0.0, "max_output_tokens": 256},
		description="Google Gemini 1.5 Pro",
	),
	"llama:llama3-70b-8192": ModelCard(
		key="llama:llama3-70b-8192",
		provider="llama",
		model="llama3-70b-8192",
		default_kwargs={"temperature": 0.0, "max_tokens": 256},
		description="Llama 3 70B via Groq",
	),
}


def list_models(registry: Optional[Dict[str, ModelCard]] = None) -> Dict[str, ModelCard]:
	return dict(DEFAULT_REGISTRY if registry is None else registry)


def create_chat_model(
	card: ModelCard,
	streaming: bool,
	request_timeout_seconds: Optional[float] = None,
	model_overrides: Optional[Dict[str, Any]] = None,
):
	builder = PROVIDER_BUILDERS.get(card.provider)
	if builder is None:
		raise ValueError(f"Unsupported provider: {card.provider}")
	kwargs: Dict[str, Any] = {**card.default_kwargs}
	if model_overrides:
		kwargs.update(model_overrides)
	return builder(card.model, streaming, request_timeout_seconds, **kwargs)

