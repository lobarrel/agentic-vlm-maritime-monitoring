#!/usr/bin/env python3
"""Maritime Traffic Monitoring — Agentic VLM Satellite PoC.

Usage:
    python main.py --lat 36.8 --lon -6.3 --timestamp 2025-03-15
    python main.py --lat 51.9 --lon 4.5                          # defaults to today
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import config
from agents import InvestigatorAgent, MonitorAgent
from image_processor import prepare_images_for_vlm

logger = logging.getLogger("vlm-maritime")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def build_report(
    monitor_report,
    investigation_report=None,
    *,
    lat: float,
    lon: float,
    timestamp: str,
) -> dict:
    """Assemble the final structured report as a JSON-serialisable dict."""
    report = {
        "target": {"lat": lat, "lon": lon, "timestamp": timestamp},
        "monitor": {
            "per_image_analysis": monitor_report.per_image_analysis,
            "temporal_summary": monitor_report.temporal_summary,
            "anomaly_detected": monitor_report.anomaly_detected,
            "anomaly_description": monitor_report.anomaly_description,
        },
    }

    if investigation_report:
        report["investigation"] = {
            "findings": investigation_report.findings,
            "evidence_chain": investigation_report.evidence_chain,
            "correlation": investigation_report.correlation,
            "skipped_directions": investigation_report.skipped_directions,
        }

    return report


def run(lat: float, lon: float, timestamp: str) -> dict:
    """Execute the full pipeline: fetch → process → monitor → investigate → report."""

    # -- Step 1: Fetch and prepare imagery -----------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 — Fetching Sentinel-2 imagery")
    logger.info("  Target: (%.4f, %.4f)  Timestamp: %s", lat, lon, timestamp)
    logger.info("=" * 60)

    images = prepare_images_for_vlm(lat, lon, timestamp)

    if not images:
        logger.error("No imagery available. Aborting.")
        return {
            "target": {"lat": lat, "lon": lon, "timestamp": timestamp},
            "error": "No cloud-free Sentinel-2 imagery found within the search window.",
        }

    logger.info("Prepared %d image(s):", len(images))
    for img in images:
        logger.info("  %s  date=%s  cloud=%.1f%%", img.path, img.date, img.cloud_cover or 0)

    # -- Step 2: Monitor Agent -----------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2 — Running Monitor Agent")
    logger.info("=" * 60)

    monitor = MonitorAgent()
    monitor_report = monitor.analyse(images)

    logger.info("Monitor complete.  Anomaly detected: %s", monitor_report.anomaly_detected)
    if monitor_report.anomaly_detected:
        logger.info("Anomaly: %s", monitor_report.anomaly_description)

    # -- Step 3: Investigator Agent (conditional) ----------------------------
    investigation_report = None
    if monitor_report.anomaly_detected:
        logger.info("=" * 60)
        logger.info("STEP 3 — Running Investigator Agent")
        logger.info("=" * 60)

        investigator = InvestigatorAgent(
            monitor_report=monitor_report,
            lat=lat,
            lon=lon,
            timestamp=timestamp,
        )
        investigation_report = investigator.investigate()

        logger.info(
            "Investigation complete.  %d finding(s) recorded.",
            len(investigation_report.findings),
        )
    else:
        logger.info("No anomaly detected — skipping investigation.")

    # -- Step 4: Report ------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4 — Generating Report")
    logger.info("=" * 60)

    report = build_report(
        monitor_report,
        investigation_report,
        lat=lat,
        lon=lon,
        timestamp=timestamp,
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maritime Traffic Monitoring — Agentic VLM Satellite PoC",
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the target location")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the target location")
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="ISO date or datetime (default: today)",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not config.OLLAMA_API_KEY and config.OLLAMA_HOST == "https://ollama.com":
        logger.error(
            "OLLAMA_API_KEY is required for Ollama Cloud.  "
            "Set it via:  export OLLAMA_API_KEY=your-key-here"
        )
        return

    config.get_client()

    report = run(args.lat, args.lon, args.timestamp)

    report_json = json.dumps(report, indent=2, default=str)

    print("\n" + "=" * 60)
    print("MARITIME TRAFFIC MONITORING REPORT")
    print("=" * 60)
    print(report_json)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report_json)
        logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
