## APISpeedTest

Latency benchmarking for LLM APIs via LangChain. Supports OpenAI, Azure OpenAI, Anthropic, Gemini, and Llama (via Groq). Easily extensible via model cards.

### Install

```bash
pip install -e .
```

### Environment Variables

- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (or pass `azure_endpoint`), `OPENAI_API_VERSION` (or pass `openai_api_version`)
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY` (for Gemini)
- `GROQ_API_KEY` (for Llama 3 via Groq)

### Quick Start

Run a quick benchmark for all default models using the default prompt:

```bash
apispeedtest --all
```

Benchmark specific models:

```bash
apispeedtest -m openai:gpt-4o-mini,anthropic:claude-3-5-sonnet-latest
```

Provide a custom prompt and number of runs:

```bash
apispeedtest --prompt "Summarize the following text: ..." --runs 5
```

Choose mode: `both` (default), `nonstream`, or `stream`:

```bash
apispeedtest --mode both
```

Save results:

```bash
apispeedtest --json-out results.json --csv-out results.csv
```

List available models:

```bash
apispeedtest list-models
```

### Config File

You can also drive the CLI with a YAML config:

```yaml
# examples/basic.yaml
prompt: |
  Explain the significance of the Fibonacci sequence in mathematics and nature.
runs: 3
mode: both  # both | nonstream | stream
models:
  - openai:gpt-4o-mini
  - azure:gpt-4o-mini
  - anthropic:claude-3-5-sonnet-latest
  - gemini:gemini-1.5-pro
  - llama:llama3-70b-8192
model_overrides:
  openai:gpt-4o-mini:
    temperature: 0.2
    max_tokens: 256
  azure:gpt-4o-mini:
    azure_endpoint: "https://YOUR-RESOURCE-NAME.openai.azure.com"
    openai_api_version: "2024-02-01"
    # api_key: "${AZURE_OPENAI_API_KEY}"
```

Run with config:

```bash
apispeedtest --config examples/basic.yaml
```

### MCP Server

This project also exposes an MCP server (via FastMCP) so you can call tools from Cursor, Claude Desktop, or any MCP client.

Install (editable) with the MCP dependency and run the server over stdio:

```bash
pip install -e .
apispeedtest-mcp
```

Available tools:

- `list_models_tool()`: returns available models and metadata
- `run_benchmark(...)`: runs benchmarks and returns structured JSON results

Example Cursor configuration (`~/.config/Cursor/mcp.json`):

```json
{
  "mcpServers": {
    "apispeedtest": {
      "command": "apispeedtest-mcp",
      "transport": "stdio",
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}",
        "GROQ_API_KEY": "${GROQ_API_KEY}"
      }
    }
  }
}
```

You can then invoke tools from your MCP-enabled client, e.g. call `run_benchmark` with parameters like `models`, `runs`, `mode`, and `request_timeout_seconds`.

### Input Tokens Impact Examples

Use these YAML configs to compare latency as you increase input token length while keeping responses short:

```bash
# short prompt (~tens of tokens)
apispeedtest --config examples/tokens-50.yaml

# medium prompt (~hundreds of tokens)
apispeedtest --config examples/tokens-medium.yaml

# large prompt (~thousands of tokens)
apispeedtest --config examples/tokens-large.yaml
```

Each config runs the same set of models and constrains `max_tokens` for the completion to reduce output-variance. Compare Non-stream avg, Stream TTFB, and Stream Total to see how input size affects timings.

### Add a New Model

Add a new entry to the registry in `apispeedtest/model_registry.py` by providing a new `ModelCard`. You can also override or add via YAML under `model_overrides`.

### Notes

- Llama support here uses Groq (`llama3-70b-8192`). You can change to other Groq-hosted models or add another backend.
- Streaming tests report both time-to-first-token (TTFB) and total streaming time.

### Metrics

- **Non-stream avg**: Average end-to-end time for a standard (non-streaming) request using `invoke()`.
- **NS TPS**: Non-streaming throughput in tokens/sec (completion tokens divided by non-streaming duration, aggregated over runs).
- **Stream TTFB**: Average time-to-first-token during streaming; measured from `stream()` start until the first chunk arrives.
- **Stream Total**: Average total wall-clock time from `stream()` start until the final chunk is received.
- **ST TPS**: Streaming throughput in tokens/sec (completion tokens divided by total streaming time; estimated from characters if provider usage is unavailable).

Totals printed below the table:

- **Prompt tokens**: Sum across all selected models and runs.
- **Completion tokens**: Sum across all selected models and runs.

