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
│ Image Processor  │   Windowed COG read → crop → PNG
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Monitor Agent   │   gemma4:31b vision — per-image + temporal analysis
└───────┬─────────┘
        │
   anomaly? ──no──► Report
        │
       yes
        │
        ▼
┌──────────────────┐
│ Investigator Agent│   gemma4:31b tool-calling loop
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
- Internet access to reach the Ollama Cloud API, Element84 STAC API, and
  S3-hosted COGs

## Setup

```bash
# Clone / navigate to the project directory
cd vlm-satellite-agent-2

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

This opens a browser window with:
- **Sidebar** for API key, host, target coordinates, date, and search parameters
- **Image gallery** displaying the fetched Sentinel-2 scenes with metadata
- **Monitor Agent panel** showing the full analysis and anomaly verdict
- **Investigation tabs** with step-by-step tool calls, evidence & findings,
  and a correlation view linking investigation results to the initial anomaly
- **Download button** to export the structured JSON report

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

# Use a local Ollama instance instead of cloud
OLLAMA_HOST=http://localhost:11434 python main.py --lat 36.0 --lon -5.5
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
| `VLM_MODEL`      | `gemma4:31b`                 | Model name for Ollama              |
| `VLM_OUTPUT_DIR` | `./output`                   | Directory for final reports        |
| `VLM_TEMP_DIR`   | `./tmp_images`               | Directory for downloaded imagery   |

In-code constants in `config.py`:

| Constant                      | Value | Description                              |
|-------------------------------|-------|------------------------------------------|
| `SEARCH_RADIUS_KM`           | 30    | Radius around target for image search    |
| `MAX_IMAGES`                  | 5     | Maximum Sentinel-2 images to fetch       |
| `MAX_CLOUD_COVER`            | 30    | Maximum cloud cover percentage           |
| `INVESTIGATOR_MAX_ITERATIONS` | 10    | Max tool-calling loop iterations         |

## How It Works

### 1. Image Acquisition

The STAC fetcher queries the Element84 Earth Search API for up to 5 recent
Sentinel-2 L2A scenes covering a 30 km radius around the target, filtered to
≤30% cloud cover.  It downloads only the `visual` (TCI) asset — a pre-composed
RGB Cloud-Optimized GeoTIFF at 10 m/pixel — using windowed reads to avoid
fetching the entire ~110 km tile.

### 2. Monitor Agent

The monitor sends all images to `gemma4:31b` as a temporal sequence and asks
it to:
- Count and locate vessels per image
- Assess port and anchorage activity
- Compare across dates for traffic pattern changes
- Flag anomalies (unusual clustering, sudden changes, vessels in unexpected
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

## Limitations

- **10 m resolution**: vessels smaller than ~30 m may not be detectable
- **Cloud cover**: maritime areas can have persistent cloud; some images may
  be partially occluded despite the filter
- **VLM accuracy**: `gemma4:31b` is a general-purpose model, not fine-tuned
  for maritime detection — appropriate for a PoC but not production use
- **Revisit time**: Sentinel-2 revisits every ~5 days; the 3–5 images may
  span 2–4 weeks

## Project Structure

```
config.py            Configuration constants
stac_fetcher.py      STAC search + COG windowed download
image_processor.py   Crop, normalise, save as PNG
tools.py             Investigator agent tool definitions
agents.py            Monitor and Investigator agent logic
main.py              CLI entry point and orchestration
gui.py               Streamlit GUI dashboard
```
