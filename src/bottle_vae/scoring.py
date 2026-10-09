"""Calibrate a saved VAE's error threshold using only normal validation images."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .model import BottleVAE
    from .preprocessing import EXPERIMENTS, preprocess_image
else:
    from model import BottleVAE
    from preprocessing import EXPERIMENTS, preprocess_image


def load_model(run_folder):
    """Load selected CPU weights strictly and check their saved dimensions and epoch."""
    checkpoint = torch.load(run_folder / "best_model.pt", map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    experiment = configuration["experiment"]
    resolution = configuration["input_resolution"]
    latent_dimension = configuration["latent_dimension"]
    if experiment not in EXPERIMENTS or (resolution, latent_dimension) != EXPERIMENTS[experiment]:
        raise ValueError("Saved model dimensions must match the E1, E2 or E3 configuration.")
    summary = json.loads((run_folder / "training_summary.json").read_text(encoding="utf-8"))
    if checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError("Checkpoint epoch differs from the selected epoch in the training summary.")
    torch.set_num_threads(4)
    model = BottleVAE(image_size=resolution, latent_dim=latent_dimension).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = {name: checkpoint[name] for name in ("configuration", "epoch", "validation_loss")}
    return model, metadata, summary


def score_images(model, paths, batch_size=8):
    """Return one mean squared RGB reconstruction error per actual source image."""
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            images = torch.stack([torch.from_numpy(preprocess_image(path, resolution=model.image_size))
                                  .permute(2, 0, 1).contiguous() for path in batch_paths])
            # Decode the latent mean so the score does not depend on sampled noise.
            reconstructions, _, _ = model(images, sample=False)
            scores = (images - reconstructions).square().mean(dim=(1, 2, 3))
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("The model produced a non-finite reconstruction score.")
            for path, score in zip(batch_paths, scores.tolist()):
                rows.append({"image": path.name, "source_image": str(path.resolve()),
                             "score": score})
    return rows


def choose_threshold(scores, percentile=95):
    """Use a recorded percentile of normal scores as the initial decision boundary."""
    if not scores or not 0 < percentile <= 100:
        raise ValueError("Supply normal scores and a percentile greater than 0 and at most 100.")
    return float(np.percentile(scores, percentile, method="linear"))


def calibrate_threshold(run_folder, data_root, percentile=95, batch_size=8):
    """Save normal-validation scores and a threshold before any test evaluation."""
    output_folder = run_folder / "calibration"
    if output_folder.exists():
        raise FileExistsError(f"Calibration is already recorded in {output_folder}; inspect its saved results.")
    split_path = run_folder / "data_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    names = sorted(split["validation"])
    if (not names or len(names) != len(set(names))
            or set(names).intersection(split["training"])):
        raise ValueError("Validation filenames must be unique, non-empty and separate from fitting.")
    checkpoint_path = run_folder / "best_model.pt"
    model, checkpoint, _ = load_model(run_folder)
    configuration = checkpoint["configuration"]
    experiment = configuration["experiment"]
    resolution, latent_dimension = configuration["input_resolution"], configuration["latent_dimension"]
    if experiment in ("E2", "E3") and len(names) != 42:
        raise ValueError("Calibration requires all 42 saved normal validation images.")
    source_folder = data_root / "train" / "good"
    rows = score_images(model, [source_folder / name for name in names], batch_size)
    scores = [row["score"] for row in rows]
    threshold = choose_threshold(scores, percentile)
    # Equality stays normal; only errors strictly above the threshold are flagged.
    for row in rows:
        row["flagged_as_anomaly"] = row["score"] > threshold
    flagged = [row["image"] for row in rows if row["flagged_as_anomaly"]]
    # Refuse to overwrite a recorded threshold after test results might be known.
    output_folder.mkdir(exist_ok=False)
    scores_path = output_folder / "validation_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "source_image", "score",
                                                   "flagged_as_anomaly"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "stage": f"{experiment} normal-validation calibration; test evaluation has not been performed",
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"],
        "data_split": str(split_path.resolve()),
        "data_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "source_folder": str(source_folder.resolve()),
        "input_resolution": resolution,
        "latent_dimension": latent_dimension,
        "preprocessing": "RGB; bilinear resize from original; float32 divided by 255",
        "score": f"mean squared error over all {resolution}*{resolution}*3 input and reconstructed RGB values; no KL",
        "reconstruction": "latent mean; sample=False; no weight updates",
        "percentile": percentile,
        "percentile_method": "numpy.percentile, method=linear",
        "threshold": threshold,
        "decision_rule": "flag as anomaly when score > threshold; equality stays normal",
        "validation_images": len(rows),
        "score_filename_order": "sorted saved validation filenames",
        "batch_size": batch_size,
        "device": "cpu",
        "cpu_threads": 4,
        "all_scores_finite": bool(np.isfinite(scores).all()),
        "validation_score_min": min(scores),
        "validation_score_mean": float(np.mean(scores)),
        "validation_score_max": max(scores),
        "validation_flagged_count": len(flagged),
        "validation_flagged_fraction": len(flagged) / len(rows),
        "validation_flagged_images": flagged,
        "validation_scores": str(scores_path.resolve()),
        "limitation": "Calibration uses the same normal validation split used for checkpoint selection; "
                      "its flagged fraction is not an independent estimate for new normal images.",
        "versions": {"NumPy": np.__version__, "PyTorch": str(torch.__version__)},
    }
    threshold_path = output_folder / "threshold.json"
    threshold_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{experiment} checkpoint epoch {checkpoint['epoch']}; {len(rows)} actual normal validation images")
    print(f"Score: mean squared RGB reconstruction error; latent-mean decoding")
    print(f"{percentile:g}th percentile threshold: {threshold!r}")
    print(f"Normal calibration images flagged: {len(flagged)}/{len(rows)} "
          f"({100 * len(flagged) / len(rows):.2f}%)")
    if flagged:
        print(f"Flagged filenames: {', '.join(flagged)}")
    print(f"Saved threshold and per-image scores: {output_folder.resolve()}")
    return output_folder


def main(argv=None):
    """Read calibration settings and record a threshold from the normal validation split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, default=Path("outputs/E1/full_training"))
    parser.add_argument("--data-root", type=Path, default=Path("data/bottle"))
    parser.add_argument("--percentile", type=float, default=95)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if not 0 < args.percentile <= 100 or args.batch_size < 1:
        parser.error("Percentile must be greater than 0 and at most 100; batch size must be positive.")
    calibrate_threshold(args.run_folder, args.data_root, args.percentile, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
