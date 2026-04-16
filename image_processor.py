"""Prepare Sentinel-2 imagery for VLM consumption."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import config
from stac_fetcher import bbox_from_point, download_visual_asset, search_sentinel2

logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    path: str
    date: str
    cloud_cover: float | None
    item_id: str
    bbox_wgs84: tuple[float, float, float, float]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_for_vlm(pixels: np.ndarray, out_path: str) -> None:
    """Normalise, resize, and save an (H, W, 3) array as JPEG for the VLM.

    Raw AOI crops can be huge; we downscale so the longest side is at most
    ``config.VLM_MAX_IMAGE_PX`` and save as JPEG at ``config.VLM_JPEG_QUALITY``.
    """
    if pixels.dtype != np.uint8:
        lo, hi = float(pixels.min()), float(pixels.max())
        if hi > lo:
            pixels = ((pixels - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            pixels = np.zeros_like(pixels, dtype=np.uint8)

    img = Image.fromarray(pixels, mode="RGB")

    max_px = config.VLM_MAX_IMAGE_PX
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)

    jpeg_path = str(Path(out_path).with_suffix(".jpg"))
    img.save(jpeg_path, format="JPEG", quality=config.VLM_JPEG_QUALITY)
    logger.info("Saved %s  (%dx%d, %d KB)", jpeg_path, img.width, img.height,
                Path(jpeg_path).stat().st_size // 1024)


def prepare_images_for_vlm(
    lat: float,
    lon: float,
    timestamp: str,
    radius_km: float = config.SEARCH_RADIUS_KM,
    max_items: int = config.MAX_IMAGES,
    max_cloud_cover: int = config.MAX_CLOUD_COVER,
    out_dir: str | None = None,
    filename_prefix: str = "",
) -> list[ImageData]:
    """End-to-end: search STAC, download, crop, save as PNG.

    *filename_prefix* is prepended to each saved file (e.g. ``explore_N_``) so
    callers can avoid clobbering other downloads that share date + item id.

    Returns a list of ImageData ordered newest-first.
    """
    out_dir = out_dir or config.TEMP_DIR
    _ensure_dir(out_dir)

    items = search_sentinel2(
        lat, lon, timestamp,
        radius_km=radius_km,
        max_items=max_items,
        max_cloud_cover=max_cloud_cover,
    )

    if not items:
        logger.warning("No Sentinel-2 items found for the given parameters")
        return []

    bbox = bbox_from_point(lat, lon, radius_km)
    results: list[ImageData] = []

    for item in items:
        try:
            pixels, meta = download_visual_asset(item, bbox)
        except Exception:
            logger.exception("Failed to download item %s — skipping", item.id)
            continue

        fname = f"{filename_prefix}{meta['datetime'][:10]}_{item.id}.jpg"
        out_path = str(Path(out_dir) / fname)
        _save_for_vlm(pixels, out_path)

        results.append(ImageData(
            path=out_path,
            date=meta["datetime"],
            cloud_cover=meta["cloud_cover"],
            item_id=item.id,
            bbox_wgs84=bbox,
        ))

    logger.info("Prepared %d images for VLM analysis", len(results))
    return results
