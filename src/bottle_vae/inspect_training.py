"""Plot saved VAE losses and preview reconstructions of held-out normal images."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop window.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from .model import BottleVAE
    from .preprocessing import EXPERIMENTS, preprocess_image
else:
    from model import BottleVAE
    from preprocessing import EXPERIMENTS, preprocess_image


# Read the losses already saved during training; do not run training again.
def load_history(run_folder):
    """Read recorded epoch losses without recalculating or smoothing them."""
    with (run_folder / "history.csv").open(newline="", encoding="utf-8") as stream:
        # CSV values are text, so convert them to numbers for plotting.
        history = [{name: float(value) for name, value in row.items()}
                   for row in csv.DictReader(stream)]
    if not history:
        raise ValueError("The run has no recorded epochs.")
    return history


# Plot fitting and validation losses, marking the selected checkpoint epoch.
def save_loss_curves(history, best_epoch, output_folder, experiment="E1"):
    """Plot training and validation total, reconstruction and KL losses."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    epochs = [row["epoch"] for row in history]
    # Make one panel each for total loss (R + K), reconstruction loss and KL loss.
    # These raw summed losses describe this run; resolution changes their scale.
    for axis, name, title in zip(axes, ("total", "reconstruction", "kl"),
                                 ("Total loss", "Reconstruction loss", "KL loss")):
        axis.plot(epochs, [row[f"train_{name}"] for row in history],
                  label="Training", color="#1565c0", linewidth=1.5)
        axis.plot(epochs, [row[f"validation_{name}"] for row in history],
                  label="Validation", color="#d55e00", linewidth=1.5)
        # The dashed line marks the saved best epoch, which may precede the final epoch.
        axis.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1,
                     label=f"Selected epoch {best_epoch}")
        axis.set(title=title, xlabel="Epoch", ylabel="Mean loss per image (log scale)")
        # A log scale keeps later improvements visible after the large initial drop.
        axis.set_yscale("log")
        axis.set_xlim(1, max(2, epochs[-1]))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"{experiment}: recorded training and normal-validation losses\n"
                   "Validation uses sampled latent vectors; dashed line marks the saved checkpoint",
                   fontsize=12)
    # Save the same plot as a PNG image and a PDF for use in the research article.
    paths = [output_folder / "loss_curves.png", output_folder / "loss_curves.pdf"]
    for path in paths:
        figure.savefig(path, dpi=300)
    plt.close(figure)
    return paths


# Save a labelled comparison of the original, prepared input and model reconstruction.
def save_reconstruction(image_path, reconstruction, output_folder, best_epoch,
                        resolution=128, experiment="E1"):
    """Show the original, model input and model output with labelled display sizes."""
    with Image.open(image_path) as original:
        original_rgb = original.convert("RGB")
    pixels = preprocess_image(image_path, resolution=resolution)
    # Convert [0, 1] values to 8-bit display intensities; model inputs are unchanged.
    input_view = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
    output_view = Image.fromarray(np.rint(reconstruction * 255).astype(np.uint8))
    # Keep all three full images side by side, with their actual resolutions labelled.
    panels = [(f"Original: {original_rgb.width} x {original_rgb.height}", original_rgb),
              (f"{experiment} input: {resolution} x {resolution}", input_view),
              (f"{experiment} reconstruction: {resolution} x {resolution}", output_view)]
    display_size, gap = 320, 24
    canvas = Image.new("RGB", (1056, 480), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text((24, 16), f"{image_path.name} | Held-out normal image | {experiment} checkpoint epoch {best_epoch}",
              fill="#17212b", font=ImageFont.load_default(size=22))
    draw.text((24, 50), "Full images; each panel displayed at 320 x 320 using bilinear scaling.",
              fill="#445566", font=font)
    # Display every panel at 320 x 320; this does not change the model resolution.
    for index, (label, panel) in enumerate(panels):
        x = gap + index * (display_size + gap)
        draw.text((x, 88), label, fill="#17212b", font=font)
        canvas.paste(panel.resize((display_size, display_size), Image.Resampling.BILINEAR),
                     (x, 118))
    draw.text((24, 450), "Reconstruction uses the latent mean. No crops, sharpening or per-image contrast adjustment.",
              fill="#445566", font=font)
    output_path = output_folder / f"{image_path.stem}_reconstruction.png"
    canvas.save(output_path)
    return output_path


# Inspect an existing training run using held-out normal images and its selected model.
def inspect_training(run_folder, data_root):
    """Load selected weights and save loss plots and three validation previews."""
    history = load_history(run_folder)
    split_path = run_folder / "data_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    checkpoint_path = run_folder / "best_model.pt"
    # Read the saved checkpoint and use its own image size and latent dimensions.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    experiment = configuration["experiment"]
    resolution, latent_dimension = configuration["input_resolution"], configuration["latent_dimension"]
    if experiment not in EXPERIMENTS or (resolution, latent_dimension) != EXPERIMENTS[experiment]:
        raise ValueError("Inspection requires dimensions matching the saved E1, E2 or E3 configuration.")
    best_epoch = checkpoint["epoch"]
    summary = json.loads((run_folder / "training_summary.json").read_text(encoding="utf-8"))
    # Confirm that the checkpoint is the one selected in the training summary.
    if best_epoch != summary["best_epoch"]:
        raise ValueError("Checkpoint epoch differs from the selected epoch in the training summary.")
    # For E1, select the first, middle and last sorted normal validation filenames.
    names = sorted(split["validation"])
    indices = sorted({0, (len(names) - 1) // 2, len(names) - 1})
    paths = [data_root / "train" / "good" / names[index] for index in indices]
    e1_inspection_path = None
    # E2 and E3 reuse E1's recorded images so the visual comparisons use the same sources.
    if experiment in ("E2", "E3"):
        e1_inspection_path = Path("outputs/E1/full_training/inspection/inspection_report.json")
        e1_inspection = json.loads(e1_inspection_path.read_text(encoding="utf-8"))
        paths = [Path(preview["source_image"]) for preview in e1_inspection["previews"]]
        if [path.name for path in paths] != ["001.png", "087.png", "206.png"]:
            raise ValueError("Inspect the same E1 normal images: 001.png, 087.png, 206.png.")
        if any(path.resolve().parent != (data_root / "train" / "good").resolve() for path in paths):
            raise ValueError("E1 inspection sources must belong to the actual normal training folder.")
    # All preview images must be held out from fitting; no reserved test images are used.
    if any(path.name not in split["validation"] or path.name in split["training"] for path in paths):
        raise ValueError("Each inspection image must belong to validation and be absent from fitting.")
    # Change each prepared image from HWC to CHW, then stack images into an NCHW batch.
    # H = height, W = width, C = RGB channels, N = number of images.
    images = torch.stack([torch.from_numpy(preprocess_image(path, resolution))
                          .permute(2, 0, 1).contiguous() for path in paths])
    torch.set_num_threads(4)
    # Recreate the matching architecture and load its already-trained weights on CPU.
    model = BottleVAE(image_size=resolution, latent_dim=latent_dimension).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    # Disable gradients: this inspection does not update any model weights.
    with torch.inference_mode():
        # The latent mean gives repeatable previews without random sampling.
        reconstructions, _, _ = model(images, sample=False)
    # Stop if the model produces NaN or infinity instead of valid image values.
    if not bool(torch.isfinite(reconstructions).all()):
        raise ValueError("The checkpoint produced non-finite reconstruction values.")
    output_folder = run_folder / "inspection"
    output_folder.mkdir(exist_ok=True)
    # Save figures under this run's inspection folder, leaving dataset originals intact.
    figures = save_loss_curves(history, best_epoch, output_folder, experiment)
    previews = []
    for path, reconstruction in zip(paths, reconstructions):
        # Convert each reconstruction back to HWC for displaying it as an RGB image.
        pixels = reconstruction.permute(1, 2, 0).numpy()
        previews.append({"source_image": str(path.resolve()),
                         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "in_validation_split": path.name in split["validation"],
                         "absent_from_fitting_split": path.name not in split["training"],
                         "output": str(save_reconstruction(path, pixels, output_folder,
                                                           best_epoch, resolution, experiment).resolve())})
    # Record source paths, file fingerprints, settings and image selection for traceability.
    report = {
        "stage": f"{experiment} training inspection; no defect-detection evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "input_resolution": resolution,
        "latent_dimension": latent_dimension,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": best_epoch,
        "checkpoint_validation_loss": checkpoint["validation_loss"],
        "history": str((run_folder / "history.csv").resolve()),
        "history_sha256": hashlib.sha256((run_folder / "history.csv").read_bytes()).hexdigest(),
        "epochs_recorded": len(history),
        "source_split": str(split_path.resolve()),
        "source_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "selection": "same three sources from E1 inspection" if experiment in ("E2", "E3")
                     else "first, middle and last sorted validation filenames",
        "e1_inspection_report": str(e1_inspection_path.resolve()) if e1_inspection_path else None,
        "e1_inspection_report_sha256": hashlib.sha256(e1_inspection_path.read_bytes()).hexdigest()
                                        if e1_inspection_path else None,
        "reconstruction_method": "decode the latent mean (sample=False); no weight updates",
        "model_shape_nchw": list(reconstructions.shape),
        "reconstruction_value_range": [float(reconstructions.min()), float(reconstructions.max())],
        "display": "full images at 320 x 320, bilinear; fixed [0, 1] to uint8 conversion",
        "loss_figures": [str(path.resolve()) for path in figures],
        "previews": previews,
    }
    report_path = output_folder / "inspection_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{experiment} checkpoint epoch {best_epoch}; {len(history)} recorded epochs")
    for preview in previews:
        print(f"Source: {preview['source_image']}")
        print(f"Preview: {preview['output']}")
    print(f"Figures and provenance: {output_folder.resolve()}")
    return output_folder


# Read the run and dataset folders from the command line, then start inspection.
def main(argv=None):
    """Read the saved run location and create its training-inspection figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, default=Path("outputs/E1/full_training"))
    parser.add_argument("--data-root", type=Path, default=Path("data/bottle"))
    args = parser.parse_args(argv)
    inspect_training(args.run_folder, args.data_root)
    return 0


# Start the command-line workflow only when this file is run directly.
if __name__ == "__main__":
    raise SystemExit(main())
