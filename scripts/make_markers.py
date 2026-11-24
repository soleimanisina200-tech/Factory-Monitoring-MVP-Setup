"""
Print-ready ArUco markers for worker identification (markers.enabled in settings.yaml).

    # how large must a marker be for my camera? (1080p, 90 deg lens, workers up to 8 m away)
    python scripts/make_markers.py --distance-m 8 --image-width 1920 --hfov-deg 90 --ids 1-40 --out markers
    # explicit size
    python scripts/make_markers.py --ids 1-40 --size-cm 20 --out markers

Where to put them:
  * vest: one on the CHEST and one on the BACK (the same id twice) - large markers fit there, readable from far;
  * helmet: small stickers on the sides / top - only for short distances (a few metres).
Each PNG is rendered at 300 dpi with a white quiet zone (needed for detection). Use matte, non-glossy material, sewn
or laminated flat (a wrinkled or curved marker is not read). Same marker id on all items of one worker.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.identity.markers import required_marker_cm  # noqa: E402


def parse_ids(text: str):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print-ready ArUco markers for helmets / vests")
    ap.add_argument("--ids", default="1-20")
    ap.add_argument("--dictionary", default="DICT_4X4_50", help="4x4 = biggest cells = readable farthest")
    ap.add_argument("--size-cm", type=float, default=None, help="printed marker side (without the white border)")
    ap.add_argument("--distance-m", type=float, default=None, help="farthest distance a worker should be read at")
    ap.add_argument("--image-width", type=int, default=1920, help="camera image width in pixels")
    ap.add_argument("--hfov-deg", type=float, default=90.0, help="horizontal field of view of the camera lens")
    ap.add_argument("--out", default="markers")
    args = ap.parse_args(argv)

    if args.distance_m:
        need = required_marker_cm(args.distance_m, args.image_width, args.hfov_deg)
        where = "helmet sticker" if need <= 10 else ("vest (chest + back)" if need <= 28 else "too large to wear")
        print(f"{args.image_width} px camera, {args.hfov_deg:.0f} deg view, {args.distance_m:.1f} m: marker side "
              f">= {need:.1f} cm  -> {where}")
        if need > 28:
            print("  -> a closer / higher-resolution / narrower-lens camera is needed for that distance")
        size = args.size_cm or round(need + 0.5)
    else:
        size = args.size_cm or 8.0
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    px = int(round(size / 2.54 * 300))
    border = px // 6
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = parse_ids(args.ids)
    for mid in ids:
        img = cv2.aruco.generateImageMarker(dictionary, mid, px)
        img = cv2.copyMakeBorder(img, border, border + 60, border, border, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(img, f"ID {mid}", (border, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
        cv2.imwrite(str(out / f"marker_{mid:03d}.png"), img)
    print(f"{len(ids)} markers written to {out} ({size:g} cm at 300 dpi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
