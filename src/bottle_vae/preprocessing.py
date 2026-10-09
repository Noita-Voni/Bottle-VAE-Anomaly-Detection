"""Inspect and prepare normal training images; no model training or scoring."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

# Latent dimensions document the later model plan; only resolution is used here.
EXPERIMENTS = {"E1": (128, 16), "E2": (256, 32), "E3": (512, 64)}
RESOLUTIONS = (128, 256, 512)
EXPECTED_COUNT = 209
EXPECTED_SIZE = (900, 900)
RESIZE_METHOD = "Pillow bilinear (Image.Resampling.BILINEAR)"

# Load an image as RGB, resize it, and scale its pixel values to [0, 1].
def preprocess_image(image_path, resolution=128):
    """Load an original as RGB, resize, and return float32 HxWxC values in [0, 1]."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Resolution must be one of {RESOLUTIONS}")
    with Image.open(image_path) as original:
        # Always resize directly from the original, never a previous small version.
        resized = original.convert("RGB").resize(
            (resolution, resolution), Image.Resampling.BILINEAR
        )
        return np.asarray(resized, dtype=np.float32) / np.float32(255.0)

# Check original images and report their properties and any unreadable files.
def inspect_originals(source_folder):
    """Verify and decode every file without converting its original mode or dtype."""
    report = {
        "expected_count": EXPECTED_COUNT,
        "expected_size_wh": list(EXPECTED_SIZE),
        "files_found": 0,
        "images_read": 0,
        "dimensions_wh": {},
        "colour_modes": {},
        "pixel_dtypes": {},
        "value_range": None,
        "discrepancies": [],
        "failures": [],
    }
    try:
        entries = sorted(source_folder.iterdir(), key=lambda path: path.name)
    except OSError as error:
        report["failures"].append(f"Cannot read source folder: {error}")
        return [], report

    paths = [path for path in entries if path.is_file()]
    report["files_found"] = len(paths)
    if len(paths) != EXPECTED_COUNT:
        report["discrepancies"].append(
            f"Expected {EXPECTED_COUNT} files; found {len(paths)}."
        )
    for path in entries:
        if not path.is_file():
            report["discrepancies"].append(f"Unexpected non-file entry: {path.name}")

    dimensions, modes, dtypes = Counter(), Counter(), Counter()
    minimum, maximum = None, None
    for path in paths:
        try:
            # verify() checks file integrity; reopening and load() also decodes pixels.
            with Image.open(path) as original:
                original.verify()
            with Image.open(path) as original:
                original.load()
                pixels = np.asarray(original)
                low, high = pixels.min().item(), pixels.max().item()
                dimensions[f"{original.width} x {original.height}"] += 1
                modes[original.mode] += 1
                dtypes[str(pixels.dtype)] += 1
                if original.size != EXPECTED_SIZE:
                    report["discrepancies"].append(
                        f"{path.name}: size {original.size}; expected {EXPECTED_SIZE}."
                    )
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
            report["images_read"] += 1
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            report["failures"].append(f"{path.name}: {type(error).__name__}: {error}")

    report["dimensions_wh"] = dict(dimensions)
    report["colour_modes"] = dict(modes)
    report["pixel_dtypes"] = dict(dtypes)
    if minimum is not None:
        report["value_range"] = [minimum, maximum]
    if report["images_read"] != EXPECTED_COUNT:
        report["discrepancies"].append(
            f"Expected {EXPECTED_COUNT} readable images; read {report['images_read']}."
        )
    return paths, report

# Check each prepared image's shape, data type, and pixel values.
def verify_preprocessing(paths, resolution):
    """Process one original at a time and report every unsuccessful image."""
    report = {
        "attempted": len(paths),
        "processed": 0,
        "resize_method": RESIZE_METHOD,
        "output_shape_hwc": [resolution, resolution, 3],
        "output_dtype": "float32",
        "value_range": None,
        "failures": [],
    }
    minimum, maximum = None, None
    for path in paths:
        try:
            pixels = preprocess_image(path, resolution)
            if pixels.shape != (resolution, resolution, 3):
                raise ValueError(f"Unexpected shape: {pixels.shape}")
            if pixels.dtype != np.float32:
                raise ValueError(f"Unexpected dtype: {pixels.dtype}")
            if not np.isfinite(pixels).all():
                raise ValueError("Non-finite pixel values")
            low, high = float(pixels.min()), float(pixels.max())
            if low < 0 or high > 1:
                raise ValueError(f"Values outside [0, 1]: {low}, {high}")
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
            report["processed"] += 1
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            report["failures"].append(f"{path.name}: {type(error).__name__}: {error}")
    if minimum is not None:
        report["value_range"] = [minimum, maximum]
    return report

# Save one labelled comparison of the original, resized versions, and crops.
def save_comparison(image_path, output_folder):
    """Label actual resolutions, equal-size display views, and matching rim crops."""
    with Image.open(image_path) as original:
        versions = [("Original", original.convert("RGB"))]
    for name, (resolution, _) in EXPERIMENTS.items():
        pixels = preprocess_image(image_path, resolution)
        # Convert back to display intensities only for this preview, not model input.
        preview = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
        versions.append((name, preview))

    display_size, gap = 320, 24
    canvas = Image.new("RGB", (1400, 960), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=24)
    draw.text((24, 16), f"{image_path.name} | Normal training image | Preprocessing comparison",
              fill="#17212b", font=title_font)
    draw.text((24, 54), "Each resized version starts from the original. Orange box marks the crop below.",
              fill="#17212b", font=font)

    for column, (name, version) in enumerate(versions):
        x = gap + column * (display_size + gap)
        draw.text((x, 94), f"{name}: {version.width} x {version.height} pixels",
                  fill="#17212b", font=font)
        direction = "enlarged" if version.width < display_size else "reduced"
        draw.text((x, 120), f"Displayed at 320 x 320 ({direction})", fill="#445566", font=font)
        canvas.paste(version.resize((display_size, display_size), Image.Resampling.BILINEAR),
                     (x, 152))
        # Use the same fractional upper-rim region, rounded down to pixel boundaries.
        box = (version.width * 3 // 8, version.height // 8,
               version.width * 5 // 8, version.height * 3 // 8)
        draw.rectangle((x + 120, 192, x + 200, 272), outline="#f08000", width=3)
        crop = version.crop(box)
        draw.text((x, 504), f"Rim crop: {crop.width} x {crop.height} pixels",
                  fill="#17212b", font=font)
        draw.text((x, 530), "Enlarged to 320 x 320 for display", fill="#445566", font=font)
        canvas.paste(crop.resize((display_size, display_size), Image.Resampling.NEAREST),
                     (x, 562))

    draw.text((24, 908), "Full views: bilinear display scaling. Crops: nearest-neighbour enlargement; no detail is restored.",
              fill="#17212b", font=font)
    draw.text((24, 934), "Crop region: x = 3/8 to 5/8, y = 1/8 to 3/8 of each image. These previews do not measure detection performance.",
              fill="#445566", font=font)
    output_path = output_folder / f"{image_path.stem}_comparison.png"
    canvas.save(output_path)
    return str(output_path)

# Select the first, middle, and last images and save their comparisons.
def save_previews(paths, output_folder):
    """Select the first, middle and last sorted filenames reproducibly."""
    report = {"selection": "first, middle, last sorted filenames", "files": [], "failures": []}
    output_folder.mkdir(parents=True, exist_ok=True)
    if paths:
        for index in sorted({0, (len(paths) - 1) // 2, len(paths) - 1}):
            path = paths[index]
            try:
                report["files"].append(save_comparison(path, output_folder))
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                report["failures"].append(f"{path.name}: {type(error).__name__}: {error}")
    return report

# Select the first, middle, and last images and save their comparisons.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="E1")
    parser.add_argument("--data-root", type=Path, default=Path("data/bottle"))
    parser.add_argument("--previews", action="store_true",
                        help="Save three original/128/256/512 comparisons under outputs/preprocessing")
    args = parser.parse_args(argv)
    source_folder = args.data_root / "train" / "good"
    resolution, latent_dimension = EXPERIMENTS[args.experiment]

    paths, inspection = inspect_originals(source_folder)
    preparation = verify_preprocessing(paths, resolution)
    previews = save_previews(paths, Path("outputs/preprocessing")) if args.previews else None
    failures = inspection["discrepancies"] + inspection["failures"] + preparation["failures"]
    if previews:
        failures += previews["failures"]
    report = {
        "stage": "Phase 2.1 preprocessing check; not a completed model experiment",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "source_folder": str(source_folder.resolve()),
        "input_resolution": resolution,
        "latent_dimension_for_later_model_only": latent_dimension,
        "versions": {"Pillow": pillow_version, "NumPy": np.__version__},
        "original_inspection": inspection,
        "preprocessing": preparation,
        "previews": previews,
        "normalization_example": {
            "rgb": [255, 128, 0],
            "scaled_rgb": [1.0, 128 / 255, 0.0],
            "meaning": "Colour intensities, not defect probabilities",
        },
        "status": "FAIL" if failures else "PASS",
    }
    output_folder = Path("outputs") / args.experiment
    output_folder.mkdir(parents=True, exist_ok=True)
    report_path = output_folder / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{args.experiment}: preprocessing check only; source: {source_folder.resolve()}")
    print(f"Originals: {inspection['images_read']}/{inspection['files_found']} readable files")
    print(f"Dimensions (W x H): {inspection['dimensions_wh']}")
    print(f"Modes: {inspection['colour_modes']}; dtypes: {inspection['pixel_dtypes']}")
    print(f"Original pixel range: {inspection['value_range']}")
    print(f"Prepared: {preparation['processed']}/{preparation['attempted']}; "
          f"shape: {preparation['output_shape_hwc']}; float32; range: {preparation['value_range']}")
    for failure in failures:
        print(f"  - {failure}")
    if previews:
        for path in previews["files"]:
            print(f"Preview: {path}")
    print(f"{report['status']}: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
