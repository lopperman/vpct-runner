# VPCT Benchmark Runner

A benchmarking framework for evaluating vision-language models on the **Visual Physics Comprehension Test (VPCT)**—a dataset designed to measure how well AI models understand intuitive physics from visual observations.

**[Leaderboard](https://cbrower.dev/vpct)** | **[VPCT-1 Dataset on Hugging Face](https://huggingface.co/datasets/camelCase12/vpct-1)**

## What is VPCT?

The Visual Physics Comprehension Test presents models with images from a ball-and-bucket physics simulation. Given an initial frame showing a ball, obstacles, and three buckets, models must predict which bucket the ball will eventually fall into. This tests spatial reasoning, trajectory prediction, and understanding of physical dynamics like gravity, collisions, and bouncing.

## Quick Start

### 1. Install Dependencies

```bash
pip install openai anthropic httpx
```

### 2. Download the Dataset

Get the [VPCT-1 dataset](https://huggingface.co/datasets/camelCase12/vpct-1) and place it in your data directory:

```bash
# Using huggingface-cli
huggingface-cli download camelCase12/vpct-1 --repo-type dataset --local-dir ./data
```

### 3. Set API Keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 4. Run a Benchmark

```bash
python run-vpct.py -d ./data -o ./results -m gpt-4o --runs 3
```

## Usage

```
python run-vpct.py [OPTIONS]
```

### Core Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --data-dir` | Path to VPCT dataset | `data` |
| `-o, --output-dir` | Where to save results | `out` |
| `-m, --models` | Comma-separated model slugs | Required |
| `-p, --prompt-file` | Custom prompt file (optional) | Built-in prompt |
| `--runs` | Number of evaluation runs | `1` |
| `--batch-size` | Concurrent requests per batch | `1` |
| `--subset` | Limit to first N simulations | All |

### Request Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--max-tokens` | Max response tokens | `4096` |
| `--timeout-seconds` | Request timeout | `600` |
| `--max-retries` | Retry attempts on failure | `5` |
| `--base-delay` | Initial retry delay (seconds) | `5.0` |
| `--overwrite` | Re-run existing results | `false` |

### Provider-Specific Options

| Option | Description |
|--------|-------------|
| `--thinking-budget` | Anthropic extended thinking tokens (0 = disabled) |
| `--openai-base-url` | Custom endpoint for OpenAI-compatible APIs |
| `--openai-api-key` | API key (overrides `OPENAI_API_KEY` env var) |

## Supported Models

### OpenAI

| Slug | Model | Notes |
|------|-------|-------|
| `gpt-4o` | GPT-4o | |
| `gpt-4o-mini` | GPT-4o Mini | |
| `gpt-4.1` | GPT-4.1 | |
| `gpt-4.1-mini` | GPT-4.1 Mini | |
| `o1-low/medium/high` | o1 | Reasoning effort variants |
| `o1-pro-low/medium/high` | o1 Pro | Reasoning effort variants |
| `o3-low/medium/high` | o3 | Reasoning effort variants |
| `o3-mini-low/medium/high` | o3 Mini | Reasoning effort variants |
| `o4-mini-low/medium/high` | o4 Mini | Reasoning effort variants |

### Anthropic

| Slug | Model | Thinking Budget |
|------|-------|-----------------|
| `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet | — |
| `claude-3-7-sonnet-20250219` | Claude 3.7 Sonnet | — |
| `claude-3-7-sonnet-20250219-16k` | Claude 3.7 Sonnet | 16,000 |
| `claude-3-7-sonnet-20250219-32k` | Claude 3.7 Sonnet | 32,000 |
| `claude-4-sonnet-20250514` | Claude Sonnet 4 | — |
| `claude-4-sonnet-20250514-16k` | Claude Sonnet 4 | 16,000 |
| `claude-4-sonnet-20250514-32k` | Claude Sonnet 4 | 32,000 |
| `claude-4-opus-20250514` | Claude Opus 4 | — |
| `claude-4-opus-20250514-16k` | Claude Opus 4 | 16,000 |

### Google (via OpenAI-compatible endpoint)

| Slug | Model |
|------|-------|
| `gemini-2.0-flash` | Gemini 2.0 Flash |
| `gemini-2.0-flash-thinking-exp-01-21` | Gemini 2.0 Flash Thinking |
| `gemini-2.5-pro-preview-05-06` | Gemini 2.5 Pro |
| `gemini-2.5-flash-preview-05-20` | Gemini 2.5 Flash |

## Examples

### Basic evaluation with GPT-4o

```bash
python run-vpct.py -d ./data -o ./runs/gpt-4o -m gpt-4o --runs 3
```

### Quick test with a subset

```bash
python run-vpct.py -d ./data -o ./runs/test -m gpt-4o-mini --subset 10 --batch-size 5
```

### Compare multiple models

```bash
python run-vpct.py -d ./data -o ./runs/comparison -m gpt-4o,claude-4-sonnet-20250514 --runs 3
```

### Using Gemini via Google's OpenAI-compatible API

```bash
python run-vpct.py \
  -d ./data \
  -o ./runs/gemini \
  -m gemini-2.5-pro-preview-05-06 \
  --openai-base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --openai-api-key $GEMINI_API_KEY
```

### Claude with extended thinking

```bash
python run-vpct.py -d ./data -o ./runs/claude-thinking -m claude-4-sonnet-20250514-32k
```

### Using a third-party provider (OpenRouter)

```bash
python run-vpct.py \
  -d ./data \
  -o ./runs/openrouter \
  -m gpt-4o \
  --openai-base-url https://openrouter.ai/api/v1 \
  --openai-api-key $OPENROUTER_API_KEY
```

## Output Format

Results are saved as JSON files in the output directory:

- `benchmark_results_{model}_run{N}.json` — Per-run results with individual predictions
- `benchmark_results_{model}_avg.json` — Aggregated accuracy across all runs

### Sample output structure

```json
{
  "model_name": "gpt-4o",
  "predictions": [
    {
      "simulation_id": 1,
      "initial_image_path": "data/sim_1_initial.png",
      "actual_bucket": 2,
      "predicted_bucket": 2,
      "is_correct": true,
      "model_response": "..."
    }
  ],
  "overall_accuracy": 0.73
}
```

## Adding Custom Models

Edit `model_registry.py` to add new models:

```python
MODEL_REGISTRY["my-model"] = {
    "provider": "openai",  # or "anthropic"
    "model": "actual-model-id",
    "reasoning_effort": "medium",  # optional, for reasoning models
    "thinking_budget": 16000,      # optional, for Anthropic
}
```

## Project Structure

```
vpct-runner/
├── run-vpct.py          # Main entry point
├── cli.py               # Argument parsing
├── model_registry.py    # Model configuration
├── prompt.py            # Default evaluation prompt
├── utils.py             # Retry logic and response parsing
├── vpct_dataclasses.py  # Result data structures
└── adapters/
    ├── openai_adapter.py    # OpenAI/compatible API integration
    └── claude_adapter.py    # Anthropic API integration
```

## License

MIT
