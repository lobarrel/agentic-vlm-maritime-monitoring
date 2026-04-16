"""Sentinel-2 imagery search and download via Element84 STAC API."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio
from pystac_client import Client
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import Point, box, mapping, shape

import config

logger = logging.getLogger(__name__)

_stac_client: Client | None = None


def _get_client() -> Client:
    global _stac_client
    if _stac_client is None:
        _stac_client = Client.open(config.STAC_API_URL)
    return _stac_client


def bbox_from_point(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Convert a centre point + radius in km to a WGS-84 bbox (west, south, east, north).

    Uses an equirectangular approximation which is accurate enough at the
    spatial scales we operate at (~30 km).
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat))

    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon

    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _item_date(item) -> str:
    """Extract the calendar date (YYYY-MM-DD) from a STAC item."""
    dt = item.datetime or item.properties.get("datetime")
    return str(dt)[:10]


def _mgrs_tile(item) -> str:
    """Sentinel-2 MGRS cell id from STAC properties (Earth Search: ``s2:mgrs_tile``)."""
    p = item.properties or {}
    return str(p.get("s2:mgrs_tile") or p.get("mgrs_tile") or "?")


def _item_spatial_key(item, point_ll: Point) -> tuple[float, float]:
    """Sort key: prefer scenes centred on the AOI, then lower cloud.

    Picking *only* minimum ``eo:cloud_cover`` per day can select a **different**
    MGRS tile than on other days (adjacent tiles often overlap the same bbox and
    can have slightly different cloud metadata).  That makes the “most recent”
    image look like a spatial jump even though the dates are correct.
    """
    try:
        geom = shape(item.geometry)
        if geom.is_empty:
            dist = 1e9
        else:
            dist = float(point_ll.distance(geom.centroid))
    except Exception:
        dist = 1e9
    cloud = float(item.properties.get("eo:cloud_cover", 100))
    return (dist, cloud)


def _deduplicate_by_date(
    items: list,
    max_dates: int,
    lat: float,
    lon: float,
) -> list:
    """Keep one STAC item per calendar date, chosen for spatial consistency.

    Sentinel-2 uses fixed MGRS tiles (~110 km); a search bbox often intersects
    more than one tile on the same pass.  We group by date, then pick the
    candidate whose footprint centroid is closest to *(lat, lon)*, breaking
    ties with lower ``eo:cloud_cover``.

    Returns up to *max_dates* items sorted newest-first.
    """
    point_ll = Point(lon, lat)
    by_date: dict[str, list] = {}
    for item in items:
        by_date.setdefault(_item_date(item), []).append(item)

    best_per_date: dict[str, object] = {}
    for date_key, cands in by_date.items():
        best_per_date[date_key] = min(
            cands,
            key=lambda it: _item_spatial_key(it, point_ll),
        )

    sorted_dates = sorted(best_per_date.keys(), reverse=True)[:max_dates]
    result = [best_per_date[d] for d in sorted_dates]

    for d in sorted_dates:
        it = best_per_date[d]
        logger.info(
            "Date %s → item %s  tile=%s  cloud=%s",
            d,
            getattr(it, "id", "?"),
            _mgrs_tile(it),
            (it.properties or {}).get("eo:cloud_cover"),
        )

    logger.info(
        "Deduplicated %d items → %d unique dates: %s",
        len(items), len(result), sorted_dates,
    )
    return result


def search_sentinel2(
    lat: float,
    lon: float,
    timestamp: str | datetime,
    radius_km: float = config.SEARCH_RADIUS_KM,
    max_items: int = config.MAX_IMAGES,
    max_cloud_cover: int = config.MAX_CLOUD_COVER,
) -> list:
    """Search Element84 STAC for recent Sentinel-2 L2A items.

    Returns up to *max_items* items — one per distinct calendar date —
    ordered newest-first, captured before *timestamp* with cloud cover
    below *max_cloud_cover* %.  This guarantees a true temporal sequence
    where each image represents a different satellite pass (~2-5 day cadence).
    """
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    bbox = bbox_from_point(lat, lon, radius_km)
    geom = mapping(box(*bbox))

    lookback = timestamp - timedelta(days=90)
    date_range = f"{lookback.strftime('%Y-%m-%d')}/{timestamp.strftime('%Y-%m-%d')}"

    logger.info(
        "STAC search  bbox=%s  date_range=%s  cloud<=%d%%",
        bbox, date_range, max_cloud_cover,
    )

    # Over-fetch so we have enough items to find max_items *unique dates*
    # after deduplication (each date may have 2-4 overlapping tiles).
    fetch_limit = max_items * 6

    client = _get_client()
    search = client.search(
        collections=[config.STAC_COLLECTION],
        intersects=geom,
        datetime=date_range,
        query={"eo:cloud_cover": {"lte": max_cloud_cover}},
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        max_items=fetch_limit,
    )

    raw_items = list(search.items())
    logger.info("STAC returned %d raw items", len(raw_items))

    return _deduplicate_by_date(
        raw_items, max_dates=max_items, lat=lat, lon=lon,
    )


def download_visual_asset(
    item,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict]:
    """Read the TCI (visual) asset from a STAC item, windowed to *bbox*.

    Parameters
    ----------
    item : pystac.Item
    bbox : (west, south, east, north) in EPSG:4326

    Returns
    -------
    (pixels, meta) where *pixels* is a uint8 ndarray of shape (H, W, 3)
    and *meta* contains CRS, transform, datetime, and cloud_cover.
    """
    asset = item.assets.get("visual")
    if asset is None:
        raise ValueError(f"Item {item.id} has no 'visual' asset")

    href = asset.href
    logger.info("Downloading visual asset from %s", href)

    with rasterio.open(href) as src:
        src_crs = src.crs

        # Reproject the WGS-84 bbox into the raster's native CRS (usually UTM)
        transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        left, bottom = transformer.transform(bbox[0], bbox[1])
        right, top = transformer.transform(bbox[2], bbox[3])

        window = from_bounds(left, bottom, right, top, src.transform)

        # Clamp window to the raster extent
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        data = src.read(window=window)  # shape (bands, H, W)

    pixels = np.moveaxis(data, 0, -1)  # -> (H, W, bands)
    if pixels.shape[2] > 3:
        pixels = pixels[:, :, :3]

    item_dt = item.datetime or item.properties.get("datetime")
    meta = {
        "item_id": item.id,
        "crs": str(src_crs),
        "datetime": str(item_dt),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "bbox_wgs84": bbox,
    }
    return pixels, meta
