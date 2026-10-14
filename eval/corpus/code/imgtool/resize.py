#!/usr/bin/env python3
"""Resize every jpg/png in a folder so the width is at most max_width.

Originals are never touched. Output goes to a 'resized' subfolder.
usage: python3 resize.py ~/Pictures/trip-lisbon [max_width]
"""
import sys
from pathlib import Path

from PIL import Image, ImageOps

DEFAULT_MAX_WIDTH = 1600
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def target_size(width, height, max_width):
    if width <= max_width:
        return width, height
    scale = max_width / width
    return max_width, round(height * scale)


def resize_one(src, dst_dir, max_width):
    with Image.open(src) as im:
        # honor the phone's EXIF rotation before measuring anything
        im = ImageOps.exif_transpose(im)
        new_size = target_size(im.width, im.height, max_width)
        if new_size == im.size:
            out = im.copy()
        else:
            out = im.resize(new_size, Image.LANCZOS)
        if out.mode in ("RGBA", "P") and src.suffix.lower() in (".jpg", ".jpeg"):
            out = out.convert("RGB")
        dst = dst_dir / src.name
        out.save(dst, quality=88, optimize=True)
    return dst


def main(folder, max_width):
    root = Path(folder).expanduser()
    dst_dir = root / "resized"
    dst_dir.mkdir(exist_ok=True)
    done = 0
    failed = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            resize_one(path, dst_dir, max_width)
            done += 1
        except OSError as exc:
            failed.append((path.name, str(exc)))
    print(f"resized {done} images into {dst_dir}")
    for name, reason in failed:
        print(f"  failed: {name} ({reason})", file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    width = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MAX_WIDTH
    sys.exit(main(folder, width))
