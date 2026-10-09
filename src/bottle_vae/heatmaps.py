"""Show spatial reconstruction errors for the three inspected normal images."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .model import BottleVAE
    from .preprocessing import EXPERIMENTS, preprocess_image
else:
    from model import BottleVAE
    from preprocessing import EXPERIMENTS, preprocess_image


# Reuse the three normal images already selected for reconstruction inspection.
def load_inspection_sources(run_folder):
    """Use only the three source paths already recorded by reconstruction inspection."""
    report = json.loads((run_folder / "inspection" / "inspection_report.json")
                        .read_text(encoding="utf-8"))
    # Read their original dataset paths from the saved inspection report.
    paths = [Path(preview["source_image"]).resolve() for preview in report["previews"]]
    if [path.name for path in paths] != ["001.png", "087.png", "206.png"]:
        raise ValueError("Expected the inspected images 001.png, 087.png and 206.png.")
    source_folder = Path("data/bottle/train/good").resolve()
    if any(path.parent != source_folder for path in paths):
        raise ValueError("Inspection sources must be the actual data/bottle/train/good images.")
    # Confirm that these images belong to validation and were not used for fitting.
    split = json.loads((run_folder / "data_split.json").read_text(encoding="utf-8"))
    if any(path.name not in split["validation"] or path.name in split["training"] for path in paths):
        raise ValueError("Each inspected image must be held out from fitting.")
    return paths, report


# Calculate both the error at each pixel and the overall error for each image.
def calculate_error_maps(images, reconstructions):
    """Average squared RGB differences across channels, retaining height and width."""
    # Squaring prevents positive and negative colour differences from cancelling out.
    squared_errors = (images - reconstructions).square()
    # Dimension 1 contains RGB channels: average them to get one error per pixel.
    error_maps = squared_errors.mean(dim=1)  # NCHW -> one H x W map per image.
    # Average over channels, height and width to get the image's reconstruction MSE.
    image_scores = squared_errors.mean(dim=(1, 2, 3))
    return error_maps, image_scores


# Save the input, reconstruction and coloured error map together in one labelled figure.
def save_heatmap(image_name, prepared, reconstruction, error_map, colour_max, output_folder,
                 resolution=128, experiment="E1", checkpoint_epoch=47):
    """Save three labelled panels with the same numerical error colour scale."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
    axes[0].imshow(prepared, interpolation="nearest")
    axes[0].set_title(f"Prepared RGB input ({resolution} x {resolution})")
    axes[1].imshow(reconstruction, interpolation="nearest")
    axes[1].set_title(f"Latent-mean reconstruction ({resolution} x {resolution})")
    # Use the same numerical colour limits for all three images in this configuration.
    coloured_map = axes[2].imshow(error_map, cmap="magma", vmin=0, vmax=colour_max,
                                 interpolation="nearest")
    axes[2].set_title(f"Spatial squared error ({resolution} x {resolution})")
    for axis in axes:
        axis.set_axis_off()
    # The colour bar connects each displayed colour to its squared-error value.
    figure.colorbar(coloured_map, ax=axes, shrink=0.8, pad=0.02,
                    label="Mean squared RGB difference")
    figure.suptitle(f"{experiment} | {image_name} | Held-out normal image | Checkpoint epoch {checkpoint_epoch}", fontsize=13)
    # Display scaling enlarges the existing pixels; it adds no new image detail.
    figure.supxlabel(f"Native {resolution} x {resolution} arrays enlarged for display using nearest-neighbour interpolation.\n"
                     "Heatmaps show reconstruction differences, not defect probabilities or confirmed defects.",
                     fontsize=9)
    output_path = output_folder / f"{Path(image_name).stem}_heatmap.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


# Load the saved model, calculate and check its error maps, then save figures and a report.
def make_heatmaps(run_folder):
    """Reconstruct the three recorded sources, check their errors, and save heatmaps."""
    paths, inspection = load_inspection_sources(run_folder)
    checkpoint_path = run_folder / "best_model.pt"
    threshold_path = run_folder / "calibration" / "threshold.json"
    # Use calibration records to identify the model; the maps do not apply a threshold.
    calibration = json.loads(threshold_path.read_text(encoding="utf-8"))
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    experiment = configuration["experiment"]
    resolution, latent_dimension = configuration["input_resolution"], configuration["latent_dimension"]
    if experiment not in EXPERIMENTS or (resolution, latent_dimension) != EXPERIMENTS[experiment]:
        raise ValueError("Heatmaps require dimensions matching the saved E1, E2 or E3 configuration.")
    split_path = run_folder / "data_split.json"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    # File fingerprints and settings must match the existing inspection and calibration.
    if (Path(inspection["checkpoint"]).resolve() != checkpoint_path.resolve()
            or inspection["checkpoint_epoch"] != checkpoint["epoch"]
            or calibration["checkpoint_epoch"] != checkpoint["epoch"]
            or checkpoint_hash != calibration["checkpoint_sha256"]
            or checkpoint_hash != inspection.get("checkpoint_sha256", checkpoint_hash)
            or split_hash != calibration["data_split_sha256"]
            or calibration["experiment"] != experiment
            or calibration["input_resolution"] != resolution
            or calibration["latent_dimension"] != latent_dimension):
        raise ValueError("Use the unchanged checkpoint and split recorded by inspection and calibration.")
    torch.set_num_threads(4)
    # Recreate the matching architecture and load its already-trained weights on CPU.
    model = BottleVAE(image_size=resolution, latent_dim=latent_dimension).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    # Prepare actual images, change HWC to CHW, and stack them into one NCHW batch.
    # H = height, W = width, C = RGB channels, N = number of images.
    images = torch.stack([torch.from_numpy(preprocess_image(path, resolution=resolution))
                          .permute(2, 0, 1).contiguous() for path in paths])
    # Disable gradients and decode the latent mean instead of drawing random samples.
    with torch.inference_mode():
        reconstructions, _, _ = model(images, sample=False)
        error_maps, image_scores = calculate_error_maps(images, reconstructions)
        # Averaging each spatial map should give the same overall RGB image score.
        map_means = error_maps.mean(dim=(1, 2))
    if error_maps.shape != (3, resolution, resolution) or not bool(torch.isfinite(error_maps).all()):
        raise ValueError(f"Expected three finite {resolution} x {resolution} error maps.")
    # Read the previously saved image scores for an additional consistency check.
    scores_path = run_folder / "calibration" / "validation_scores.csv"
    with scores_path.open(newline="", encoding="utf-8") as stream:
        saved_scores = {row["image"]: float(row["score"]) for row in csv.DictReader(stream)}
    # Allow tiny float32 rounding differences when comparing the two averaging orders.
    relative_tolerance, absolute_tolerance = 1e-5, 1e-8
    results = []
    # Check every map mean against both its current RGB score and its saved score.
    for index, path in enumerate(paths):
        map_mean = float(map_means[index])
        rgb_score = float(image_scores[index])
        saved_score = saved_scores[path.name]
        checks = {
            "map_mean_matches_rgb_score": bool(np.isclose(map_mean, rgb_score,
                                                         rtol=relative_tolerance, atol=absolute_tolerance)),
            "map_mean_matches_saved_score": bool(np.isclose(map_mean, saved_score,
                                                           rtol=relative_tolerance, atol=absolute_tolerance)),
        }
        if not all(checks.values()):
            raise ValueError(f"{path.name}: spatial mean does not match its image score within tolerance.")
        results.append({"source_image": str(path),
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "map_mean": map_mean, "rgb_image_score": rgb_score,
                        "saved_calibration_score": saved_score,
                        "difference_from_rgb_score": abs(map_mean - rgb_score),
                        "difference_from_saved_score": abs(map_mean - saved_score),
                        "checks": checks})
    # One absolute scale for all three images; do not normalise each map separately.
    # The largest pixel error across all three maps sets this run's shared upper limit.
    colour_max = float(error_maps.max())
    output_folder = run_folder / "heatmaps"
    output_folder.mkdir(exist_ok=True)
    # Convert CHW back to HWC for plotting and save one figure for each source image.
    for index, path in enumerate(paths):
        output_path = save_heatmap(path.name, images[index].permute(1, 2, 0).numpy(),
                                  reconstructions[index].permute(1, 2, 0).numpy(),
                                  error_maps[index].numpy(), colour_max, output_folder,
                                  resolution, experiment, checkpoint["epoch"])
        results[index]["figure"] = str(output_path.resolve())
        print(f"{path.name}: map mean={results[index]['map_mean']:.10f}; both score checks passed")
    # Record sources, model identity, score checks, colour limits and display settings.
    report = {
        "stage": f"{'Phase 3' if experiment == 'E1' else 'Phase 4'} {experiment} normal-image heatmaps; no reserved test evaluation",
        "experiment": experiment,
        "input_resolution": resolution,
        "latent_dimension": latent_dimension,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inspection_report": str((run_folder / "inspection" / "inspection_report.json").resolve()),
        "inspection_report_sha256": hashlib.sha256(
            (run_folder / "inspection" / "inspection_report.json").read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": checkpoint_hash,
        "source_split": str(split_path.resolve()), "source_split_sha256": split_hash,
        "threshold_sha256": hashlib.sha256(threshold_path.read_bytes()).hexdigest(),
        "saved_score_source": str(scores_path.resolve()),
        "saved_score_source_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "reconstruction": f"{experiment} {resolution}/{latent_dimension}; latent mean (sample=False); inference mode; no weight updates",
        "preprocessing": f"RGB; original resized bilinearly to {resolution} x {resolution}; float32 divided by 255",
        "error_map": "mean((input - reconstruction)^2) across three RGB channels at each pixel",
        "map_shape_hw": [resolution, resolution], "map_dtype": "float32",
        "tolerance": {"rtol": relative_tolerance, "atol": absolute_tolerance},
        "colour_scale": {"colormap": "magma", "vmin": 0.0, "vmax": colour_max,
                         "shared_across_all_three": True,
                         "maximum_rule": "largest spatial error across these three maps; linear scale"},
        "display": f"native {resolution} x {resolution} arrays enlarged to figure panels with nearest-neighbour interpolation; 300 dpi PNG",
        "meaning": "reconstruction differences, not defect probabilities or confirmed defect locations",
        "images": results,
    }
    (output_folder / "heatmap_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Shared error colour scale: 0 to {colour_max:.10f}")
    print(f"Heatmaps and provenance: {output_folder.resolve()}")
    return output_folder


# Read the chosen run folder from the command line and start the heatmap workflow.
def main(argv=None):
    """Read the saved run location and generate only its three inspection heatmaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, default=Path("outputs/E1/full_training"))
    args = parser.parse_args(argv)
    make_heatmaps(args.run_folder)
    return 0


# Start the command-line workflow only when this file is run directly.
if __name__ == "__main__":
    raise SystemExit(main())
