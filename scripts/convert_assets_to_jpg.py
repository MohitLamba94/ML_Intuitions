"""
Convert non-jpg raster images in `assets/` to `.jpg` to save space.

Motivation: jpg is far smaller than png for the photo-like/figure images we embed
in the notes. This is a reusable, going-forward utility: whenever a non-jpg image
is brought in (e.g. downloaded from a blog post), run this to convert it.

IMPORTANT -- existing images are left untouched. The files in PRESERVE below are
already committed and referenced by other notes (converting them would break those
links), so the script always skips them. To convert only specific new files, pass
their paths as arguments; with no arguments it scans `assets/` and converts every
non-jpg raster image that is NOT in PRESERVE.

On success it writes `<name>.jpg` (RGB, alpha flattened onto white, quality 70) and
removes the original only after the jpg is written.

Run:
    # convert all eligible non-jpg files in assets/ (skips PRESERVE)
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/convert_assets_to_jpg.py

    # convert only specific files
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/convert_assets_to_jpg.py assets/some_download.png
"""

import os
import sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))

# Raster formats we convert. (svg/pdf are vector -- not handled here.)
CONVERTIBLE_EXTS = {".png", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# Pre-existing images referenced by other notes -- never touch these.
PRESERVE = {
    "ddpm_algorithm.png",
    "probability_sharpening.png",
    "dpo_beta_loss_surface.png",
    "dpo_qualitative_results_1.png",
    "dpo_qualitative_results_2.png",
}


def convert_one(path):
    """Convert a single image file to a sibling .jpg; return the new path or None."""
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    if ext.lower() in (".jpg",):
        print("skip (already jpg):", base)
        return None
    if ext.lower() not in CONVERTIBLE_EXTS:
        print("skip (not a convertible raster):", base)
        return None
    if base in PRESERVE:
        print("skip (preserved / referenced by other notes):", base)
        return None

    out = os.path.join(os.path.dirname(path), stem + ".jpg")
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))  # flatten alpha onto white
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.save(out, "JPEG", quality=70, optimize=True)

    os.remove(path)  # only reached if save succeeded
    print(f"converted: {base} -> {os.path.basename(out)}")
    return out


def main(argv):
    if argv:
        targets = [os.path.abspath(p) for p in argv]
    else:
        targets = [
            os.path.join(ASSETS, f)
            for f in sorted(os.listdir(ASSETS))
            if os.path.splitext(f)[1].lower() in CONVERTIBLE_EXTS
        ]
    n = 0
    for t in targets:
        if convert_one(t):
            n += 1
    print(f"done ({n} converted)")


if __name__ == "__main__":
    main(sys.argv[1:])
