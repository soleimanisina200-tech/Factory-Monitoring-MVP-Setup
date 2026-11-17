"""
Print-ready ArUco markers for helmet identification (markers.enabled in settings.yaml).

    python scripts/make_markers.py --ids 1-40 --size-cm 8 --out markers

Each PNG is rendered at 300 dpi with a white quiet zone (needed for detection). Use matte (non-glossy) stickers;
put one on each side of the helmet so it is visible from several directions.
"""
import argparse
from pathlib import Path

import cv2


def parse_ids(text: str):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="1-20")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--size-cm", type=float, default=8.0, help="printed marker side (without the white border)")
    ap.add_argument("--out", default="markers")
    args = ap.parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    px = int(round(args.size_cm / 2.54 * 300))
    border = px // 6
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for mid in parse_ids(args.ids):
        img = cv2.aruco.generateImageMarker(dictionary, mid, px)
        img = cv2.copyMakeBorder(img, border, border + 60, border, border, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(img, f"ID {mid}", (border, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
        cv2.imwrite(str(out / f"marker_{mid:03d}.png"), img)
    print(f"{len(parse_ids(args.ids))} markers written to {out} ({args.size_cm} cm at 300 dpi)")


if __name__ == "__main__":
    main()
