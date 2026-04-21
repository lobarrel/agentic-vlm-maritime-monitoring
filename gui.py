#!/usr/bin/env python3
"""Streamlit GUI for the Maritime Traffic Monitoring PoC.

Launch with:
    streamlit run gui.py
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import date

import streamlit as st

import config
from agents import InvestigatorAgent, MonitorAgent
from image_processor import ImageData, prepare_images_for_vlm
from session_snapshot import (
    apply_manifest_to_session,
    build_snapshot_zip,
    load_snapshot_from_zip_bytes,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Maritime Traffic Monitor",
    page_icon="satellite",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* tighten default padding */
    .block-container { padding-top: 1.5rem; }

    /* image gallery cards */
    .img-card {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .img-card img { border-radius: 4px; }
    .img-card .meta {
        font-size: 0.78rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: 4px;
    }

    /* tool call badges */
    .tool-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: monospace;
    }
    .tool-explore_direction        { background: #dbeafe; color: #1e40af; }
    .tool-skip_direction           { background: #f3f4f6; color: #6b7280; }
    .tool-analyze_image            { background: #ede9fe; color: #5b21b6; }
    .tool-submit_finding           { background: #d1fae5; color: #065f46; }

    /* confidence pills */
    .confidence-high   { background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 10px; font-size: 0.78rem; font-weight: 600; }
    .confidence-medium { background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 10px; font-size: 0.78rem; font-weight: 600; }
    .confidence-low    { background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 10px; font-size: 0.78rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "images": None,
    "monitor_report": None,
    "investigation_report": None,
    "inv_steps": [],
    "pipeline_done": False,
    "error": None,
    "lat": 29.89,
    "lon": 32.54,
    "radius_km": config.SEARCH_RADIUS_KM,
    "max_cloud": config.MAX_CLOUD_COVER,
    "max_images": config.MAX_IMAGES,
}
DEFAULT_SNAPSHOT_DATE = date(2021, 3, 29)

# Keys cleared when a new analysis starts.  Must NOT include sidebar widget
# keys (``lat``, ``lon``, ``snap_ts``, sliders): those are locked after the
# widgets render, and ``run_pipeline`` runs after the sidebar.
_PIPELINE_STATE_DEFAULTS = {
    "images": None,
    "monitor_report": None,
    "investigation_report": None,
    "inv_steps": [],
    "pipeline_done": False,
    "error": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "snap_ts" not in st.session_state:
    st.session_state["snap_ts"] = DEFAULT_SNAPSHOT_DATE

# Apply snapshot before any widget with keys ``lat`` / ``snap_ts`` / … is
# created; otherwise Streamlit forbids writing those session_state keys.
_pending_bytes = st.session_state.pop("pending_snapshot_bytes", None)
_pending_digest = st.session_state.pop("pending_snapshot_digest", None)
if _pending_bytes is not None:
    try:
        manifest, root = load_snapshot_from_zip_bytes(_pending_bytes)
        apply_manifest_to_session(manifest, root, st.session_state)
        if _pending_digest is not None:
            st.session_state["_loaded_snap_digest"] = _pending_digest
        st.session_state["snapshot_flash_ok"] = True
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        st.session_state["snapshot_flash_err"] = str(exc)


def _reset_session() -> None:
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default
    st.session_state["snap_ts"] = DEFAULT_SNAPSHOT_DATE
    st.session_state.pop("_loaded_snap_digest", None)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Maritime Traffic Monitor")
    st.caption("Agentic VLM Satellite PoC")

    st.markdown("---")
    st.subheader("Ollama Cloud")

    api_key = st.text_input(
        "API Key",
        value=st.session_state.get("api_key", config.OLLAMA_API_KEY),
        type="password",
        help="Generate at https://ollama.com/settings/keys",
    )
    st.session_state["api_key"] = api_key

    ollama_host = st.text_input(
        "Host",
        value=st.session_state.get("ollama_host", config.OLLAMA_HOST),
        help="Ollama Cloud: https://ollama.com  /  Local: http://localhost:11434",
    )
    st.session_state["ollama_host"] = ollama_host

    st.markdown("---")
    st.subheader("Target Location")

    lat = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        step=0.1,
        format="%.4f",
        key="lat",
    )
    lon = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        step=0.1,
        format="%.4f",
        key="lon",
    )
    st.date_input("Date (search before)", key="snap_ts")

    st.markdown("---")
    st.subheader("Settings")

    st.slider("Search radius (km)", 5, 60, key="radius_km")
    st.slider("Max cloud cover (%)", 0, 80, key="max_cloud")
    st.slider("Max images", 1, 8, key="max_images")

    st.markdown("---")

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if st.session_state.pipeline_done:
        st.button("Reset", on_click=_reset_session, use_container_width=True)

    st.markdown("---")
    st.subheader("Saved dashboard")
    st.caption(
        "Save or restore the **exact** Streamlit view (sidebar + results). "
        "The .zip includes images so you can move it to another machine."
    )
    snap_upload = st.file_uploader(
        "Load snapshot",
        type=["zip"],
        help="Choose a file previously downloaded with “Download dashboard (.zip)”.",
        key="snap_upload",
    )
    if snap_upload is not None:
        raw = snap_upload.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        if st.session_state.get("_loaded_snap_digest") != digest:
            st.session_state["pending_snapshot_bytes"] = raw
            st.session_state["pending_snapshot_digest"] = digest
            st.rerun()

    st.markdown("---")
    st.caption(f"Model: `{config.MODEL_NAME}`")
    st.caption(f"STAC: `{config.STAC_COLLECTION}`")

timestamp = st.session_state["snap_ts"]
radius_km = int(st.session_state["radius_km"])
max_cloud = int(st.session_state["max_cloud"])
max_images = int(st.session_state["max_images"])


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def render_image_gallery(images: list[ImageData]) -> None:
    """Show fetched images in a responsive grid."""
    n = len(images)
    cols = st.columns(min(n, 5))
    for i, img in enumerate(images):
        col = cols[i % len(cols)]
        with col:
            st.image(img.path, use_container_width=True)
            date_short = img.date[:10] if img.date else "?"
            cloud_str = f"{img.cloud_cover:.0f}%" if img.cloud_cover is not None else "n/a"
            st.caption(f"{date_short}  \u2601 {cloud_str}")


def render_tool_badge(tool_name: str) -> str:
    return f'<span class="tool-badge tool-{tool_name}">{tool_name}</span>'


def render_confidence(level: str) -> str:
    return f'<span class="confidence-{level}">{level.upper()}</span>'


def render_investigation_step(step: dict) -> None:
    """Render a single investigator step inside a status container."""
    event = step["event"]
    data = step["data"]

    if event == "investigator_thinking":
        it = data["iteration"]
        mx = data["max_iterations"]
        st.markdown(f"**Iteration {it}/{mx}** — thinking...")

    elif event == "investigator_reasoning":
        with st.expander(f"Reasoning (iteration {data['iteration']})", expanded=False):
            st.markdown(data["content"])

    elif event == "tool_call":
        badge = render_tool_badge(data["tool"])
        args_str = ", ".join(f"{k}={v!r}" for k, v in data["arguments"].items())
        st.markdown(
            f"{badge} `({args_str})`",
            unsafe_allow_html=True,
        )

    elif event == "tool_result":
        with st.expander(f"Result from {data['tool']}", expanded=True):
            result_text = data["result"]
            if any(
                result_text.lower().endswith(ext)
                for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
            ):
                st.image(result_text, use_container_width=True)
            else:
                st.code(result_text, language="text")

    elif event == "investigator_correlating":
        st.markdown("**Correlating findings with the original anomaly...**")


def render_findings(findings: list[dict]) -> None:
    """Render each finding as a card with evidence images."""
    if not findings:
        st.info("The investigator did not record any formal findings.")
        return

    for i, f in enumerate(findings, 1):
        conf_html = render_confidence(f.get("confidence", "medium"))
        st.markdown(
            f"### Finding {i}: {f['title']}  {conf_html}",
            unsafe_allow_html=True,
        )
        st.markdown(f["description"])

        evidence = f.get("evidence_images", [])
        if evidence:
            ev_cols = st.columns(min(len(evidence), 4))
            for j, img_path in enumerate(evidence):
                if os.path.isfile(img_path):
                    ev_cols[j % len(ev_cols)].image(img_path, use_container_width=True)
                else:
                    ev_cols[j % len(ev_cols)].caption(f"`{img_path}` (not found)")

        if i < len(findings):
            st.markdown("---")


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(
    lat: float,
    lon: float,
    timestamp_str: str,
    radius_km: float,
    max_cloud: int,
    max_images: int,
) -> None:
    """Run the full pipeline, writing results to session state and updating
    Streamlit containers live as steps complete."""

    st.session_state.update(_PIPELINE_STATE_DEFAULTS)

    # -- Step 1: fetch imagery -----------------------------------------------
    with st.status("Fetching Sentinel-2 imagery...", expanded=True) as status:
        st.write(f"Searching {config.STAC_API_URL}")
        st.write(f"Centre: ({lat:.4f}, {lon:.4f})  Radius: {radius_km} km  Cloud <= {max_cloud}%")

        images = prepare_images_for_vlm(
            lat, lon, timestamp_str,
            radius_km=radius_km,
            max_items=max_images,
            max_cloud_cover=max_cloud,
        )

        if not images:
            status.update(label="No imagery found", state="error")
            st.session_state["error"] = (
                "No cloud-free Sentinel-2 imagery found within the 90-day "
                "search window.  Try increasing the cloud cover threshold "
                "or changing the date."
            )
            return

        st.write(f"Retrieved **{len(images)}** image(s)")
        status.update(label=f"Fetched {len(images)} images", state="complete")

    st.session_state["images"] = images

    # -- Step 2: monitor agent -----------------------------------------------
    monitor_placeholder = st.empty()

    with monitor_placeholder.status("Monitor Agent analysing imagery...", expanded=True) as status:
        reasoning_slot = st.empty()

        def _monitor_cb(event: str, data: dict) -> None:
            if event == "monitor_complete":
                reasoning_slot.markdown(data["raw_response"])

        monitor = MonitorAgent()
        report = monitor.analyse(images, on_step=_monitor_cb)
        st.session_state["monitor_report"] = report

        if report.anomaly_detected:
            status.update(label="Monitor complete — anomaly detected", state="complete")
        else:
            status.update(label="Monitor complete — no anomaly", state="complete")

    # -- Step 3: investigator agent (conditional) ----------------------------
    if report.anomaly_detected:
        inv_steps: list[dict] = []
        st.session_state["inv_steps"] = inv_steps

        with st.status("Investigator Agent running...", expanded=True) as status:
            step_slot = st.container()

            def _inv_cb(event: str, data: dict) -> None:
                step = {"event": event, "data": data}
                inv_steps.append(step)
                with step_slot:
                    render_investigation_step(step)

            investigator = InvestigatorAgent(
                monitor_report=report,
                lat=lat,
                lon=lon,
                timestamp=timestamp_str,
            )
            inv_report = investigator.investigate(on_step=_inv_cb)
            st.session_state["investigation_report"] = inv_report

            n = len(inv_report.findings)
            status.update(
                label=f"Investigation complete — {n} finding(s)",
                state="complete",
            )

    st.session_state["pipeline_done"] = True


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

st.markdown("## Maritime Traffic Monitoring")
st.markdown(
    "Analyse Sentinel-2 satellite imagery for maritime traffic patterns "
    "and anomalies using an agentic VLM pipeline."
)

_snap_err = st.session_state.pop("snapshot_flash_err", None)
if _snap_err:
    st.error(f"Could not load snapshot: {_snap_err}")
if st.session_state.pop("snapshot_flash_ok", False):
    st.success("Dashboard restored — same layout as when you saved it.")

# Start the pipeline when the button is clicked
if run_btn:
    key = st.session_state.get("api_key", "")
    host = st.session_state.get("ollama_host", config.OLLAMA_HOST)

    if not key and host == "https://ollama.com":
        st.error(
            "An API key is required for Ollama Cloud.  "
            "Enter your key in the sidebar or set the `OLLAMA_API_KEY` "
            "environment variable."
        )
    else:
        config.get_client(api_key=key, host=host)
        ts_str = str(timestamp)
        run_pipeline(lat, lon, ts_str, radius_km, max_cloud, max_images)

# --- Render persisted results from session state ---

if st.session_state["error"]:
    st.error(st.session_state["error"])

images = st.session_state["images"]
monitor_report = st.session_state["monitor_report"]
investigation_report = st.session_state["investigation_report"]

# -- Image gallery -----------------------------------------------------------
if images:
    st.markdown("---")
    st.markdown("### Sentinel-2 Imagery")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Images", len(images))
    dates = [img.date[:10] for img in images if img.date]
    if dates:
        metric_cols[1].metric("Date range", f"{dates[-1]} — {dates[0]}")
    avg_cloud = sum(i.cloud_cover or 0 for i in images) / len(images)
    metric_cols[2].metric("Avg cloud", f"{avg_cloud:.0f}%")
    metric_cols[3].metric("Resolution", "10 m/px")

    render_image_gallery(images)

# -- Monitor report ----------------------------------------------------------
if monitor_report:
    st.markdown("---")
    st.markdown("### Monitor Agent Analysis")

    if monitor_report.anomaly_detected:
        st.warning(f"Anomaly detected: {monitor_report.anomaly_description}")
    else:
        st.success("No anomaly detected in the observed imagery.")

    with st.expander("Full monitor reasoning", expanded=False):
        st.markdown(monitor_report.raw_response)

# -- Investigation -----------------------------------------------------------
if investigation_report:
    st.markdown("---")
    st.markdown("### Investigation")

    tab_steps, tab_findings, tab_correlation = st.tabs([
        "Agent Steps", "Evidence & Findings", "Correlation",
    ])

    # ---- Steps tab ---------------------------------------------------------
    with tab_steps:
        inv_steps = st.session_state.get("inv_steps", [])
        if inv_steps:
            for step in inv_steps:
                render_investigation_step(step)
        else:
            st.info("No investigation steps recorded.")

    # ---- Findings tab ------------------------------------------------------
    with tab_findings:
        render_findings(investigation_report.findings)

    # ---- Correlation tab ---------------------------------------------------
    with tab_correlation:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Initial Anomaly")
            st.info(monitor_report.anomaly_description or "No description")
        with col_b:
            st.markdown("#### Investigation Conclusion")
            if investigation_report.correlation:
                st.success(investigation_report.correlation)
            else:
                st.info("No correlation summary available.")

        if investigation_report.evidence_chain:
            st.markdown("#### Evidence Chain")
            st.markdown(investigation_report.evidence_chain)

        if investigation_report.skipped_directions:
            st.markdown("#### Directions Not Explored")
            for s in investigation_report.skipped_directions:
                st.markdown(f"- **{s['direction']}**: {s['reason']}")

# -- Download report ---------------------------------------------------------
if st.session_state["pipeline_done"]:
    st.markdown("---")
    report_dict = {
        "target": {"lat": lat, "lon": lon, "timestamp": str(timestamp)},
    }
    if monitor_report:
        report_dict["monitor"] = {
            "per_image_analysis": monitor_report.per_image_analysis,
            "temporal_summary": monitor_report.temporal_summary,
            "anomaly_detected": monitor_report.anomaly_detected,
            "anomaly_description": monitor_report.anomaly_description,
        }
    if investigation_report:
        report_dict["investigation"] = {
            "findings": investigation_report.findings,
            "evidence_chain": investigation_report.evidence_chain,
            "correlation": investigation_report.correlation,
            "skipped_directions": investigation_report.skipped_directions,
        }

    report_json = json.dumps(report_dict, indent=2, default=str)

    zip_bytes = build_snapshot_zip(
        lat=float(lat),
        lon=float(lon),
        timestamp=timestamp,
        radius_km=radius_km,
        max_cloud=max_cloud,
        max_images=max_images,
        images=images if images else None,
        monitor_report=monitor_report,
        investigation_report=investigation_report,
        inv_steps=st.session_state.get("inv_steps") or None,
        pipeline_done=st.session_state["pipeline_done"],
        error=st.session_state.get("error"),
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download JSON report",
            data=report_json,
            file_name="maritime_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="Download dashboard (.zip)",
            data=zip_bytes,
            file_name="maritime_dashboard.zip",
            mime="application/zip",
            help="Reopen this app later and use “Load snapshot” in the sidebar. "
            "Restores the full dashboard, not a static web page.",
            use_container_width=True,
        )
