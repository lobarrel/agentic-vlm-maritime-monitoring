"""Save / restore the Streamlit dashboard state (exact UI replay in-app)."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

import config
from agents import InvestigationReport, MonitorReport
from image_processor import ImageData

MANIFEST = "manifest.json"
ASSETS = "assets"
VERSION = 1


def _collect_paths(
    images: list[ImageData] | None,
    investigation_report: InvestigationReport | None,
    inv_steps: list[dict[str, Any]] | None,
) -> list[str]:
    out: list[str] = []
    if images:
        for img in images:
            if img.path:
                out.append(os.path.abspath(img.path))
    if investigation_report and investigation_report.findings:
        for f in investigation_report.findings:
            for p in f.get("evidence_images") or []:
                if isinstance(p, str) and p:
                    out.append(os.path.abspath(p))
    if inv_steps:
        for step in inv_steps:
            if step.get("event") != "tool_result":
                continue
            r = (step.get("data") or {}).get("result")
            if isinstance(r, str) and any(
                r.lower().endswith(ext)
                for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
            ):
                if os.path.isfile(r):
                    out.append(os.path.abspath(r))
    seen: set[str] = set()
    unique: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _rewrite_paths_in_inv_steps(
    inv_steps: list[dict[str, Any]],
    mapping: dict[str, str],
) -> list[dict[str, Any]]:
    def rewrite_str(s: str) -> str:
        for old, new in mapping.items():
            if old in s:
                s = s.replace(old, new)
        return s

    out: list[dict[str, Any]] = []
    for step in inv_steps:
        step = json.loads(json.dumps(step, default=str))
        if step.get("event") == "tool_result":
            data = step.get("data") or {}
            r = data.get("result")
            if isinstance(r, str):
                data["result"] = rewrite_str(r)
                step["data"] = data
        out.append(step)
    return out


def _rewrite_findings_paths(
    findings: list[dict[str, Any]] | None,
    mapping: dict[str, str],
) -> list[dict[str, Any]] | None:
    if not findings:
        return findings
    fixed: list[dict[str, Any]] = []
    for f in json.loads(json.dumps(findings, default=str)):
        ev = f.get("evidence_images")
        if ev:
            new_ev = []
            for p in ev:
                if isinstance(p, str):
                    np = p
                    for old, new in mapping.items():
                        if old in np:
                            np = np.replace(old, new)
                    new_ev.append(np)
                else:
                    new_ev.append(p)
            f["evidence_images"] = new_ev
        fixed.append(f)
    return fixed


def build_snapshot_zip(
    *,
    lat: float,
    lon: float,
    timestamp: date | datetime | str,
    radius_km: float,
    max_cloud: int,
    max_images: int,
    images: list[ImageData] | None,
    monitor_report: MonitorReport | None,
    investigation_report: InvestigationReport | None,
    inv_steps: list[dict[str, Any]] | None,
    pipeline_done: bool,
    error: str | None,
) -> bytes:
    """Return a zip bytes blob with manifest + copied image assets."""

    if isinstance(timestamp, datetime):
        ts_str = timestamp.date().isoformat()
    elif isinstance(timestamp, date):
        ts_str = timestamp.isoformat()
    else:
        ts_str = str(timestamp)[:10]

    all_paths = _collect_paths(images, investigation_report, inv_steps)
    mapping: dict[str, str] = {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, abs_path in enumerate(all_paths):
            base = os.path.basename(abs_path) or f"file_{i}"
            arcname = f"{ASSETS}/{i:03d}_{base}"
            mapping[abs_path] = arcname
            if os.path.isfile(abs_path):
                zf.write(abs_path, arcname)

        images_payload: list[dict[str, Any]] = []
        if images:
            for img in images:
                ap = os.path.abspath(img.path) if img.path else ""
                rel = mapping.get(ap)
                if not rel and img.path and os.path.isfile(img.path):
                    ap = os.path.abspath(img.path)
                    rel = f"{ASSETS}/{len(mapping):03d}_{os.path.basename(img.path)}"
                    mapping[ap] = rel
                    zf.write(img.path, rel)
                d = asdict(img)
                if rel:
                    d["path"] = rel
                images_payload.append(d)

        inv_for_manifest = _rewrite_paths_in_inv_steps(
            list(inv_steps or []),
            dict(mapping),
        )
        inv_report_dict = None
        if investigation_report:
            inv_report_dict = asdict(investigation_report)
            inv_report_dict["findings"] = _rewrite_findings_paths(
                inv_report_dict.get("findings"),
                dict(mapping),
            )

        manifest: dict[str, Any] = {
            "version": VERSION,
            "pipeline_done": pipeline_done,
            "error": error,
            "ui": {
                "lat": lat,
                "lon": lon,
                "timestamp": ts_str,
                "radius_km": radius_km,
                "max_cloud": max_cloud,
                "max_images": max_images,
            },
            "images": images_payload,
            "monitor_report": asdict(monitor_report) if monitor_report else None,
            "investigation_report": inv_report_dict,
            "inv_steps": inv_for_manifest,
        }
        zf.writestr(MANIFEST, json.dumps(manifest, indent=2, default=str))
    return buf.getvalue()


def _resolve_asset_path(asset_rel: str, extract_root: str) -> str:
    return os.path.normpath(os.path.join(extract_root, asset_rel.replace("/", os.sep)))


def _looks_like_windows_abs(s: str) -> bool:
    """True for ``C:\\foo`` / ``C:/foo`` — on POSIX, :func:`os.path.isabs` is False for these."""
    s = s.strip()
    return len(s) >= 3 and s[1] == ":" and s[2] in "\\/"


def _find_asset_by_basename(extract_root: str, want_basename: str) -> str | None:
    """Locate ``want_basename`` under ``assets/``.

    Zip entries use ``{ASSETS}/{i:03d}_{original_basename}``; manifests may
    still list a stale absolute path whose basename matches ``original_basename``.
    """
    if not want_basename:
        return None
    assets_dir = os.path.join(extract_root, ASSETS)
    if not os.path.isdir(assets_dir):
        return None
    for name in os.listdir(assets_dir):
        path = os.path.join(assets_dir, name)
        if not os.path.isfile(path):
            continue
        if name == want_basename:
            return os.path.normpath(path)
        if len(name) >= 4 and name[:3].isdigit() and name[3] == "_":
            if name[4:] == want_basename:
                return os.path.normpath(path)
    return None


def resolve_manifest_path(stored: str | None, extract_root: str) -> str:
    """Turn a path stored in a manifest into a path that exists under *extract_root*.

    Handles:

    - Relative paths such as ``assets/001_foo.jpg`` (normal export).
    - Stale **absolute** paths from the machine where the zip was built; the
      file is looked up by basename inside ``assets/`` (older exports / reuse
      of a downloaded zip on another host).
    - Absolute paths that embed ``.../assets/...`` (try the tail).
    """
    if not isinstance(stored, str):
        return ""
    s = stored.strip()
    if not s:
        return ""
    if os.path.isfile(s):
        return os.path.normpath(os.path.abspath(s))
    stale_abs = os.path.isabs(s) or _looks_like_windows_abs(s)
    # Typical case: relative arcname in manifest (e.g. assets/001_foo.jpg)
    if not stale_abs:
        rel = _resolve_asset_path(s, extract_root)
        if os.path.isfile(rel):
            return rel
    else:
        # Absolute or Windows path but missing (wrong machine / cleaned temp dir)
        norm = s.replace("\\", "/")
        if "/assets/" in norm:
            tail = norm.split("/assets/", 1)[-1].lstrip("/")
            if tail:
                rel = _resolve_asset_path(tail, extract_root)
                if os.path.isfile(rel):
                    return rel
    want = os.path.basename(s.replace("\\", "/"))
    found = _find_asset_by_basename(extract_root, want)
    if found:
        return found
    # Best-effort relative join for odd manifests; stale absolute → keep string for UI
    if not stale_abs:
        return _resolve_asset_path(s, extract_root)
    return s


def load_snapshot_from_zip_bytes(data: bytes) -> tuple[dict[str, Any], str]:
    """Extract zip to a temp directory and return ``(manifest, extract_root)``."""

    root = tempfile.mkdtemp(prefix="maritime_snap_")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if MANIFEST not in names and f"./{MANIFEST}" not in names:
            shutil.rmtree(root, ignore_errors=True)
            raise ValueError(
                f"Missing {MANIFEST}: not a valid dashboard snapshot export."
            )
        zf.extractall(root)
    manifest_path = os.path.join(root, MANIFEST)
    if not os.path.isfile(manifest_path):
        shutil.rmtree(root, ignore_errors=True)
        raise ValueError(f"Missing {MANIFEST} after extract.")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest, root


def apply_manifest_to_session(
    manifest: dict[str, Any],
    extract_root: str,
    session_state: Any,
) -> None:
    """Populate Streamlit session_state from a loaded manifest."""

    old = session_state.get("snapshot_extract_dir")
    if isinstance(old, str) and os.path.isdir(old) and old != extract_root:
        try:
            shutil.rmtree(old)
        except OSError:
            pass

    ui = manifest.get("ui") or {}
    ts = ui.get("timestamp", "")
    try:
        y, m, d = (int(x) for x in str(ts)[:10].split("-"))
        session_state["snap_ts"] = date(y, m, d)
    except (ValueError, TypeError):
        session_state["snap_ts"] = datetime.now().date()

    session_state["lat"] = float(ui.get("lat", 29.92))
    session_state["lon"] = float(ui.get("lon", 32.54))
    session_state["radius_km"] = int(ui.get("radius_km", config.SEARCH_RADIUS_KM))
    session_state["max_cloud"] = int(ui.get("max_cloud", config.MAX_CLOUD_COVER))
    session_state["max_images"] = int(ui.get("max_images", config.MAX_IMAGES))

    session_state["pipeline_done"] = bool(manifest.get("pipeline_done"))
    session_state["error"] = manifest.get("error")

    imgs_raw = manifest.get("images") or []
    images: list[ImageData] = []
    for row in imgs_raw:
        p = row.get("path")
        full = resolve_manifest_path(p, extract_root) if isinstance(p, str) and p else str(p or "")
        bbox = row.get("bbox_wgs84")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_t = tuple(float(x) for x in bbox)
        else:
            bbox_t = (0.0, 0.0, 0.0, 0.0)
        images.append(
            ImageData(
                path=full,
                date=str(row.get("date", "")),
                cloud_cover=row.get("cloud_cover"),
                item_id=str(row.get("item_id", "")),
                bbox_wgs84=bbox_t,
            )
        )
    session_state["images"] = images if images else None

    mr = manifest.get("monitor_report")
    session_state["monitor_report"] = (
        MonitorReport(**mr) if isinstance(mr, dict) and mr else None
    )

    ir = manifest.get("investigation_report")
    if isinstance(ir, dict) and ir:
        findings = ir.get("findings") or []
        if findings:
            fixed_findings = []
            for f in findings:
                fc = dict(f)
                ev = fc.get("evidence_images") or []
                fc["evidence_images"] = [
                    resolve_manifest_path(ep, extract_root)
                    if isinstance(ep, str)
                    else ep
                    for ep in ev
                ]
                fixed_findings.append(fc)
            ir = {**ir, "findings": fixed_findings}
        session_state["investigation_report"] = InvestigationReport(**ir)
    else:
        session_state["investigation_report"] = None

    steps = manifest.get("inv_steps") or []
    fixed_steps: list[dict[str, Any]] = []
    for step in steps:
        step = json.loads(json.dumps(step))
        if step.get("event") == "tool_result":
            data = step.get("data") or {}
            r = data.get("result")
            if isinstance(r, str) and any(
                r.lower().endswith(ext)
                for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
            ):
                if r:
                    data["result"] = resolve_manifest_path(r, extract_root)
                step["data"] = data
        fixed_steps.append(step)
    session_state["inv_steps"] = fixed_steps

    session_state["snapshot_extract_dir"] = extract_root
