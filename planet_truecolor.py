"""Weekly Planet (PlanetScope, ~3m) true color images of the camp for the Streamlit app.

Each run:
  1. Searches PlanetScope scenes over a 2km box around the camp from the last 45 days
     (the Education & Research plan has a 30-day delay on new imagery).
  2. Orders and downloads the sharpest new scene that is >=95% clear over the box
     (at most one per run; each scene uses ~100 km2 of the 3000 km2/month quota).
  3. Renders any downloaded scene not yet in S3, brightened, as a 2km view
     (images/planet_<date>.png -> s3 planet/<date>.png) and a 600m camp view
     (images/planet_point_<date>.png -> s3 planet_point/<date>.png).

Planet computes the per-box cloud check asynchronously, so scenes whose check isn't ready
are left for the next run. Needs PLANET_LABS_KEY in .env.

cd /Users/andrewwoodill/Repos/penage-satellite-maps/ && source env/bin/activate && python planet_truecolor.py
"""

import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape

DATA_API = "https://api.planet.com/data/v1"
ORDERS_API = "https://api.planet.com/compute/ops/orders/v2"
ITEM_TYPE = "PSScene"
BUNDLE = "visual"

BUCKET = "penage-true-color-images"
# S3 prefix -> (local image name, crop half-width in m around the camp; None = full 2km box)
VIEWS = {
    "planet/": ("planet", None),
    "planet_point/": ("planet_point", 300),
}

# 2km x 2km box centered here
CENTER_LAT, CENTER_LON = 46.19787, -81.20211
BUFFER_M = 1000
# Same camp location as proc_daily_point() in senthub_truecolor.py
CAMP_LON, CAMP_LAT = -81.20123776310807, 46.198907336078946

SINCE_DAYS = 45
MAX_SCENE_CLOUD = 0.1
MIN_AOI_CLEAR = 95
MIN_VALID = 0.9  # fraction of the box with image data (clip corners can be slightly empty)
MAX_SHIFT_M = 30  # some PlanetScope scenes are misregistered by 100m+
MAX_CANDIDATES = 10

DATA_DIR = Path("planet_data")
STATE_FILE = DATA_DIR / "downloaded.json"
# A well-registered clip (2026-07-29) that new clips are compared against
REFERENCE = DATA_DIR / "reference.tif"


def read_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def bbox_polygon(lat, lon, buffer_m):
    dlat = buffer_m / 111_320
    dlon = buffer_m / (111_320 * math.cos(math.radians(lat)))
    ring = [
        [lon - dlon, lat - dlat],
        [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat],
        [lon - dlon, lat + dlat],
        [lon - dlon, lat - dlat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def search(s, aoi, start, exclude):
    filters = [
        {"type": "GeometryFilter", "field_name": "geometry", "config": aoi},
        {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": MAX_SCENE_CLOUD}},
        {"type": "PermissionFilter", "config": ["assets:download"]},
        {"type": "DateRangeFilter", "field_name": "acquired", "config": {"gte": f"{start}T00:00:00Z"}},
    ]
    body = {"item_types": [ITEM_TYPE], "filter": {"type": "AndFilter", "config": filters}}
    r = s.post(f"{DATA_API}/quick-search", params={"_page_size": 250, "_sort": "acquired desc"}, json=body)
    r.raise_for_status()
    # The AOI clear check only counts the part of the box a scene covers, so require full coverage
    box = shape(aoi)
    features = [f for f in r.json()["features"]
                if f["id"] not in exclude and shape(f["geometry"]).contains(box)]
    # Sharpest first, then least cloud, then most recent
    features.sort(key=lambda f: (
        f["properties"].get("gsd", 99),
        f["properties"].get("cloud_cover", 1),
        -datetime.fromisoformat(f["properties"]["acquired"][:19]).timestamp(),
    ))
    return features


def aoi_clear_percent(s, item_id, aoi):
    """Percent of the AOI that is clear (UDM2), or None while Planet is still computing it. Free."""
    r = s.post(f"{DATA_API}/item-types/{ITEM_TYPE}/items/{item_id}/coverage",
               params={"mode": "UDM2"}, json={"geometry": aoi})
    r.raise_for_status()
    result = r.json()
    return result.get("clear_percent") if result.get("status") == "complete" else None


def order_and_download(s, item_id, aoi):
    name = f"penage_{item_id}"
    body = {
        "name": name,
        "products": [{"item_ids": [item_id], "item_type": ITEM_TYPE, "product_bundle": BUNDLE}],
        "tools": [{"clip": {"aoi": aoi}}],
    }
    r = s.post(ORDERS_API, json=body)
    r.raise_for_status()
    order_id = r.json()["id"]
    print(f"Placed order {order_id} for {item_id}")

    deadline = time.time() + 3600
    while time.time() < deadline:
        order = s.get(f"{ORDERS_API}/{order_id}").json()
        if order["state"] in ("success", "partial"):
            break
        if order["state"] in ("failed", "cancelled"):
            raise RuntimeError(f"Order {order_id} {order['state']}: {order.get('error_hints')}")
        time.sleep(15)
    else:
        raise RuntimeError(f"Timed out waiting for order {order_id}")

    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in order["_links"].get("results", []):
        dest = out_dir / Path(result["name"]).name
        with s.get(result["location"], stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"Downloaded {dest}")


def stretch(rgba, low=0, high=99.9, gamma=0.9):
    """Brighten: map the low-high percentile of all bands (over valid pixels) to 0-255, then lift
    shadows with a gamma. One range for all bands keeps the natural color balance."""
    rgb = rgba[..., :3].astype(float)
    lo, hi = np.percentile(rgb[rgba[..., 3] > 0], [low, high])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1), 0, 1) ** gamma * 255
    out = rgba.copy()
    out[..., :3] = rgb.astype(np.uint8)
    return out


def render_png(tif_path, png_path, half_width_m=None):
    """Render a brightened PNG; half_width_m crops to a box around the camp."""
    tif = Image.open(tif_path)
    tags = tif.tag_v2
    geokeys = tags[34735]
    epsg = {geokeys[i]: geokeys[i + 3] for i in range(4, len(geokeys), 4)}[3072]
    _, _, _, x0, y0, _ = tags[33922]
    sx, sy, _ = tags[33550]
    rgba = stretch(np.array(tif.convert("RGBA")))

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = to_utm.transform(CAMP_LON, CAMP_LAT)
    px, py = (cx - x0) / sx, (y0 - cy) / sy

    if half_width_m:
        r = int(half_width_m / sx)
        left, top = int(px) - r, int(py) - r
        rgba = rgba[max(top, 0):top + 2 * r, max(left, 0):left + 2 * r]

    h, w = rgba.shape[:2]
    dpi = 100
    scale = 1300 / max(h, w)  # same output size for both views
    fig = plt.figure(figsize=(w * scale / dpi, h * scale / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgba, interpolation="lanczos" if half_width_m else "nearest")
    ax.set_axis_off()
    fig.savefig(png_path, dpi=dpi, format="png")
    plt.close(fig)


def water_mask(tif_path):
    rgba = np.array(Image.open(tif_path).convert("RGBA")).astype(float)
    valid = rgba[..., 3] > 0
    gray = rgba[..., :3].mean(-1)
    gray[~valid] = np.median(gray[valid])
    water = (gray < np.median(gray)).astype(float)
    return water - water.mean()


def shift_m(tif_path, reference=REFERENCE):
    """Offset in metres between this clip and the reference, by phase correlation of water masks."""
    ref, img = water_mask(reference), water_mask(tif_path)
    window = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    cross = np.fft.fft2(ref * window) * np.conj(np.fft.fft2(img * window))
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-9)).real
    dy, dx = np.unravel_index(corr.argmax(), corr.shape)
    h, w = corr.shape
    dy, dx = (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)
    return 3 * math.hypot(dx, dy)


def fetch_new_scene(s, aoi):
    seen = set(json.loads(STATE_FILE.read_text())) if STATE_FILE.exists() else set()
    start = (datetime.today() - timedelta(days=SINCE_DAYS)).strftime("%Y-%m-%d")

    # Scene ids start with YYYYMMDD; one image per day is enough
    have_dates = {scene_id[:8] for scene_id in seen}
    candidates = [f for f in search(s, aoi, start, seen) if f["id"][:8] not in have_dates]

    pending = 0
    for f in candidates[:MAX_CANDIDATES]:
        clear = aoi_clear_percent(s, f["id"], aoi)
        if clear is None:
            pending += 1
        elif clear >= MIN_AOI_CLEAR:
            print(f"{f['id']}: {clear}% clear, gsd={f['properties'].get('gsd')}m")
            order_and_download(s, f["id"], aoi)
            DATA_DIR.mkdir(exist_ok=True)
            STATE_FILE.write_text(json.dumps(sorted(seen | {f["id"]}), indent=2))
            return
    print(f"No new clear scene to download ({pending} still being checked)")


def publish(s, aoi, s3_client):
    published = {}
    for prefix in VIEWS:
        listing = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        published[prefix] = {Path(o["Key"]).stem for o in listing.get("Contents", [])}

    # date -> (clear %, tif), keeping the clearest scene per date
    best = {}
    for tif in sorted(DATA_DIR.glob("*/*_Visual_clip.tif")):
        scene_id = tif.name.split("_3B")[0]
        date = f"{scene_id[:4]}-{scene_id[4:6]}-{scene_id[6:8]}"
        if all(date in published[prefix] for prefix in VIEWS):
            continue
        valid = (np.array(Image.open(tif).convert("RGBA"))[..., 3] > 0).mean()
        if valid < MIN_VALID:
            print(f"{scene_id}: only covers {valid:.0%} of the box, not publishing")
            continue
        offset = shift_m(tif)
        if offset > MAX_SHIFT_M:
            print(f"{scene_id}: misregistered by ~{offset:.0f}m, not publishing")
            continue
        clear = aoi_clear_percent(s, scene_id, aoi)
        if clear is None:
            print(f"{scene_id}: cloud check not ready, will retry next run")
        elif clear < MIN_AOI_CLEAR:
            print(f"{scene_id}: only {clear}% clear, not publishing")
        elif date not in best or clear > best[date][0]:
            best[date] = (clear, tif)

    for date, (clear, tif) in sorted(best.items()):
        for prefix, (name, half_width_m) in VIEWS.items():
            if date in published[prefix]:
                continue
            png = f"images/{name}_{date}.png"
            render_png(tif, png, half_width_m)
            s3_client.upload_file(Filename=png, Bucket=BUCKET, Key=f"{prefix}{date}.png")
            print(f"Saved: s3://{BUCKET}/{prefix}{date}.png")


if __name__ == "__main__":
    read_env()
    s = requests.Session()
    s.auth = (os.environ["PLANET_LABS_KEY"], "")
    aoi = bbox_polygon(CENTER_LAT, CENTER_LON, BUFFER_M)

    session = boto3.Session(profile_name="personal")
    s3_client = session.client("s3")

    print(f"=== planet_truecolor {datetime.now():%Y-%m-%d %H:%M} ===")
    fetch_new_scene(s, aoi)
    publish(s, aoi, s3_client)
