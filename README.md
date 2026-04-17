# Maritime Traffic Monitoring — Agentic VLM Satellite PoC

A proof-of-concept demonstrating how agentic vision-language models can run
onboard a satellite to monitor maritime traffic from Sentinel-2 imagery.

The system fetches recent satellite images, analyses them for ship positions
and traffic patterns, detects anomalies, and — when something unusual is
found — autonomously launches an investigation that explores surrounding areas
to explain the anomaly.

## Architecture

```
Input (lat, lon, timestamp)
        │
        ▼
┌───────────────┐     Element84 STAC API
│  STAC Fetcher │────────────────────────► sentinel-2-l2a
└───────┬───────┘     (pystac-client)       (COG visual asset)
        │
        ▼
┌─────────────────┐
│ Image Processor │   Windowed COG read → crop → PNG
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Monitor Agent  │   qwen3.5 vision — per-image + temporal analysis
└───────┬─────────┘
        │
   anomaly? ──no──► Report
        │
       yes
        │
        ▼
┌──────────────────┐
│ Investigator Agent │    qwen3.5 tool-calling loop
│                    │   Tools: explore_direction, skip_direction,
│                    │          analyze_image, submit_finding
└───────┬────────────┘
        │
        ▼
      Report (JSON)
```

## Prerequisites

- **Python 3.10+**
- **Ollama Cloud API key** — generate one at https://ollama.com/settings/keys

## Setup

```bash
# Clone / navigate to the project directory
cd agentic-vlm-maritime-monitoring

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your Ollama Cloud API key
export OLLAMA_API_KEY="your-key-here"
```

## Usage

### GUI (Streamlit)

The recommended way to use the application — a full dashboard showing fetched
imagery, agent reasoning, investigation steps, evidence, and correlation:

```bash
streamlit run gui.py
```

You can enter your API key directly in the sidebar, or pre-set it via the
environment variable.  The sidebar also lets you switch the host to a local
Ollama instance if needed.

### CLI

```bash
# Set your API key first
export OLLAMA_API_KEY="your-key-here"

# Basic: analyse maritime traffic near the Strait of Gibraltar
python main.py --lat 36.0 --lon -5.5 --timestamp 2025-03-15

# With debug logging
python main.py --lat 36.0 --lon -5.5 --timestamp 2025-03-15 -v

# Save the report to a file
python main.py --lat 51.9 --lon 4.5 --output report.json

# Defaults to today's date if --timestamp is omitted
python main.py --lat 36.0 --lon -5.5

```

### CLI Arguments

| Flag          | Required | Description                                  |
|---------------|----------|----------------------------------------------|
| `--lat`       | yes      | Latitude of the target location              |
| `--lon`       | yes      | Longitude of the target location             |
| `--timestamp` | no       | ISO date/datetime (default: today)           |
| `--output`    | no       | Path to save the JSON report                 |
| `-v`          | no       | Enable verbose debug logging                 |

## Configuration

All tuneable parameters live in `config.py` and can be overridden via
environment variables:

| Variable         | Default                      | Description                        |
|------------------|------------------------------|------------------------------------|
| `OLLAMA_API_KEY` | *(empty)*                    | Ollama Cloud API key (required)    |
| `OLLAMA_HOST`    | `https://ollama.com`         | Ollama endpoint (cloud or local)   |
| `VLM_MODEL`      | `qwen3.5`                    | Model name for Ollama              |
| `VLM_OUTPUT_DIR` | `./output`                   | Directory for final reports        |
| `VLM_TEMP_DIR`   | `./tmp_images`               | Directory for downloaded imagery   |

In-code constants in `config.py`:

| Constant                      | Value | Description                              |
|-------------------------------|-------|------------------------------------------|
| `SEARCH_RADIUS_KM`            | 15    | Radius around target for image search    |
| `MAX_IMAGES`                  | 4     | Maximum Sentinel-2 images to fetch       |
| `MAX_CLOUD_COVER`             | 30    | Maximum cloud cover percentage           |
| `INVESTIGATOR_MAX_ITERATIONS` | 10    | Max tool-calling loop iterations         |

## How It Works

### 1. Image Acquisition

The STAC fetcher queries the Element84 Earth Search API for recent
Sentinel-2 L2A scenes covering a specified radius around the target.  
It downloads only the `visual` (TCI) asset — a pre-composed
RGB Cloud-Optimized GeoTIFF at 10 m/pixel — using windowed reads to avoid
fetching the entire ~110 km tile.

### 2. Monitor Agent

The monitor sends all images to the VLM as a temporal sequence and asks
it to analyze maritime activity, compare across dates for traffic pattern changes
and flag anomalies (unusual clustering, sudden changes, vessels in unexpected
locations)

The model returns structured analysis ending with an anomaly verdict.

### 3. Investigator Agent

If an anomaly is detected, the investigator agent is spawned with the anomaly
context and access to these tools:

- **explore_direction**: fetch and analyse imagery adjacent to the anomaly
  along a compass bearing
- **skip_direction**: record that a direction was not explored, with reasoning
- **analyze_image**: ask the VLM a focused question about a specific image
- **submit_finding**: record a conclusion with evidence and confidence level

The agent runs a tool-calling loop (up to 10 iterations), autonomously deciding
which tools to invoke and in what order.  After submitting findings, it provides
a final correlation summary linking its evidence to the original anomaly.

### 4. Report

The output is a structured JSON report containing the monitor assessment,
any investigation findings, and the correlation analysis.

The full analysis can also be downloaded as a .zip file and can be later uploaded
to restore the dashboard and visually inspect the results.

## Limitations

- **10 m resolution**: vessels smaller than ~30 m may not be detectable
- **Cloud cover**: maritime areas can have persistent cloud; some images may
  be partially occluded despite the filter
- **VLM accuracy**: Ollama provides general-purpose models, not fine-tuned
  for maritime detection — appropriate for a PoC but not production use
- **Revisit time**: Sentinel-2 revisits every 2-5 days; the fetched images may
  span across weeks