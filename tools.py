"""Tool functions exposed to the Investigator Agent.

Each function's docstring and type annotations are automatically converted to
a JSON-schema tool descriptor by the Ollama Python SDK.  The agent loop in
agents.py dispatches tool_calls to the functions registered in TOOL_REGISTRY.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import config
from image_processor import prepare_images_for_vlm

logger = logging.getLogger(__name__)

# Accumulated findings written by the investigator via submit_finding()
findings: list[dict] = []

# Directions the investigator explicitly chose not to explore, with reasons.
skipped_directions: list[dict] = []

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

_BEARING_DEG = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315,
}


def _offset_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Return (lat, lon) at *distance_km* along *bearing_deg* from the origin."""
    R = 6371.0
    d = distance_km / R
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) +
        math.cos(lat1) * math.sin(d) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# Anchor point and anomaly context set by the investigator agent at init time.
_anchor_lat: float = 0.0
_anchor_lon: float = 0.0
_anchor_timestamp: str = ""
_anomaly_description: str = ""


def set_anchor(lat: float, lon: float, timestamp: str, anomaly: str = "") -> None:
    """Set the spatial/temporal anchor and anomaly context for exploration."""
    global _anchor_lat, _anchor_lon, _anchor_timestamp, _anomaly_description
    _anchor_lat = lat
    _anchor_lon = lon
    _anchor_timestamp = timestamp
    _anomaly_description = anomaly


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def explore_direction(
    direction: str,
    distance_km: float = 10.0,
    radius_km: float = 5.0,
) -> str:
    """Fetch and analyse a satellite image of an adjacent area in a given
    compass direction from the anomaly location.

    The tool fetches the most recent image, then automatically analyses it
    for maritime activity that could be related to the detected anomaly.
    It returns both the image path and a detailed visual analysis.

    Args:
        direction: Compass direction to explore — one of N, NE, E, SE, S, SW, W, NW.
        distance_km: How far from the anomaly centre to look (default 10).
        radius_km: Radius of the area to fetch around the target point (default 5).
    """
    direction = direction.upper().strip()
    bearing = _BEARING_DEG.get(direction)
    if bearing is None:
        return f"Error: invalid direction '{direction}'. Use one of: {', '.join(_BEARING_DEG)}"

    target_lat, target_lon = _offset_point(_anchor_lat, _anchor_lon, bearing, distance_km)

    logger.info(
        "explore_direction  dir=%s  dist=%skm  target=(%.4f, %.4f)",
        direction, distance_km, target_lat, target_lon,
    )

    ts = _anchor_timestamp or datetime.now(timezone.utc).isoformat()
    images = prepare_images_for_vlm(
        lat=target_lat,
        lon=target_lon,
        timestamp=ts,
        radius_km=radius_km,
        max_items=1,
        filename_prefix=f"explore_{direction}_",
    )

    if not images:
        return (
            f"No cloud-free imagery found {direction} of the anomaly at "
            f"({target_lat:.4f}, {target_lon:.4f})."
        )

    img = images[0]
    header = (
        f"Area {direction} of anomaly "
        f"(centre {target_lat:.4f}, {target_lon:.4f}, {distance_km} km away)\n"
        f"Image: {img.path}  (date={img.date}, cloud={img.cloud_cover}%)\n"
    )

    analysis_prompt = (
        f"You are examining a Sentinel-2 satellite image taken {direction} "
        f"of a maritime anomaly location ({distance_km} km away).\n\n"
        f"The detected anomaly is: {_anomaly_description}\n\n"
        f"Analyse this image thoroughly:\n"
        f"1. Is this area water, a canal, land, coastline, or a mix?\n"
        f"2. Count and describe any vessels visible (size, position, "
        f"   wakes, formations, direction of travel).\n"
        f"3. Note any port infrastructure, anchorages, or mooring areas.\n"
        f"4. Identify anything that could EXPLAIN or be CORRELATED with "
        f"   the anomaly described above (e.g. a fleet heading toward the "
        f"   anomaly area, a congested port pushing traffic outward, "
        f"   unusual vessel clustering, oil spill traces, etc.).\n"
        f"5. If nothing relevant is visible, say so clearly."
    )

    logger.info("Auto-analysing explored image %s", img.path)
    response = config.get_client().chat(
        model=config.MODEL_NAME,
        messages=[{
            "role": "user",
            "content": analysis_prompt,
            "images": [img.path],
        }],
    )

    return f"{header}\nVISUAL ANALYSIS:\n{response.message.content}"


def skip_direction(
    direction: str,
    reason: str,
) -> str:
    """Record a decision to NOT explore a given direction, with justification.

    Call this tool when you determine that a direction is not worth
    investigating.  This keeps the investigation transparent and focused.

    Args:
        direction: Compass direction being skipped — one of N, NE, E, SE, S, SW, W, NW.
        reason: Brief explanation of why this direction is not relevant.
    """
    direction = direction.upper().strip()
    entry = {"direction": direction, "reason": reason}
    skipped_directions.append(entry)
    logger.info("Skipping direction %s: %s", direction, reason)
    return f"Noted: skipping {direction} — {reason}"


def analyze_image(image_path: str, question: str) -> str:
    """Analyse a satellite image with a focused question using the VLM.

    Use this tool when you need a detailed visual analysis of a specific
    image.  Provide a clear, specific question about what to look for
    (e.g. ship counts, port activity, wake patterns, unusual objects).

    Args:
        image_path: Path to the image file to analyse.
        question: A specific analytical question about the image contents.
    """
    logger.info("analyze_image  path=%s  question=%s", image_path, question)

    response = config.get_client().chat(
        model=config.MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [image_path],
            }
        ],
    )
    return response.message.content


def submit_finding(
    title: str,
    description: str,
    evidence_images: str = "",
    confidence: str = "medium",
) -> str:
    """Record an investigative finding with supporting evidence.

    Call this tool when you have reached a conclusion about one aspect of
    the anomaly.  You may call it multiple times for separate findings.
    Describe the spatial relationship between the evidence and the anomaly.

    Args:
        title: Short title summarising the finding.
        description: Detailed explanation including the direction/area where
                     the evidence was found and how it correlates with the anomaly.
        evidence_images: Comma-separated paths to images that support the finding.
        confidence: Confidence level — one of "low", "medium", "high".
    """
    finding = {
        "title": title,
        "description": description,
        "evidence_images": [p.strip() for p in evidence_images.split(",") if p.strip()],
        "confidence": confidence,
    }
    findings.append(finding)
    logger.info("Finding recorded: %s  (confidence=%s)", title, confidence)
    return (
        f"Finding recorded: '{title}' (confidence={confidence}).  "
        f"Total findings so far: {len(findings)}."
    )


# Registry mapping function names to callables — used by the agent loop.
TOOL_REGISTRY: dict[str, callable] = {
    "explore_direction": explore_direction,
    "skip_direction": skip_direction,
    "analyze_image": analyze_image,
    "submit_finding": submit_finding,
}
