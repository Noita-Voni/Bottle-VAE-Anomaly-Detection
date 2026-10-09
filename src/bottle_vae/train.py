"""Train E1, E2 or E3 using only normal bottle images, with held-out validation."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .loss import vae_loss
    from .model import BottleVAE
    from .preprocessing import EXPERIMENTS, preprocess_image
else:  # Allow the same direct-file commands used in earlier phases.
    from loss import vae_loss
    from model import BottleVAE
    from preprocessing import EXPERIMENTS, preprocess_image


# E2/E3 use this saved E1 split by default so all configurations use the same images.
E1_SPLIT_FILE = Path("outputs/E1/full_training/data_split.json")


# Let PyTorch request prepared images one at a time without saving a resized dataset.
class NormalBottleDataset(Dataset):
    """Read originals and preprocess each image when it is requested."""

    # Store the source filenames and the chosen input resolution.
    def __init__(self, paths, resolution=128):
        self.paths = paths
        self.resolution = resolution

    # Tell the data loader how many images are available.
    def __len__(self):
        return len(self.paths)

    # Read and prepare one original image when the data loader requests it.
    def __getitem__(self, index):
        pixels = preprocess_image(self.paths[index], resolution=self.resolution)
        # Change HWC to CHW: height/width/RGB channels becomes channels/height/width.
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


# Divide the 209 normal training images into 167 fitting and 42 validation images.
def split_normal_images(source_folder, seed):
    """Split only train/good reproducibly; never move or copy dataset files."""
    paths = sorted(source_folder.iterdir())
    unexpected = [path.name for path in paths
                  if not path.is_file() or path.suffix.lower() != ".png"]
    if unexpected:
        raise ValueError(f"Unexpected training entries: {unexpected}")
    if len(paths) != 209:
        raise ValueError(f"Expected 209 normal training PNGs; found {len(paths)}.")
    # Shuffle only the filename list with a fixed seed; dataset files stay in place.
    random.Random(seed).shuffle(paths)
    # Use 80% for fitting, rounding down; the remaining images are held out for validation.
    split_index = int(len(paths) * 0.8)
    return paths[:split_index], paths[split_index:]


# Reuse an existing fitting/validation split, checking its membership and filename order.
def load_saved_split(source_folder, split_file):
    """Check E1's saved filename lists and reuse their exact order without reshuffling."""
    manifest_bytes = split_file.read_bytes()
    split = json.loads(manifest_bytes.decode("utf-8"))
    training, validation = split["training"], split["validation"]
    if not isinstance(training, list) or not isinstance(validation, list):
        raise ValueError("Saved training and validation entries must be filename lists.")
    names = training + validation
    if any(not isinstance(name, str) or Path(name).name != name
           or Path(name).suffix.lower() != ".png" for name in names):
        raise ValueError("The saved split must contain PNG filenames without directories.")
    entries = list(source_folder.iterdir())
    actual_names = {path.name for path in entries}
    # Every actual image must appear once, with no overlap between fitting and validation.
    checks = {
        "fitting_count_167": len(training) == 167,
        "validation_count_42": len(validation) == 42,
        "unique_filenames": len(set(names)) == len(names),
        "zero_overlap": not (set(training) & set(validation)),
        "source_has_209_normal_pngs": len(entries) == 209 and all(
            path.is_file() and path.suffix.lower() == ".png" for path in entries),
        "membership_matches_source": set(names) == actual_names,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Saved split check failed: {', '.join(failures)}")
    # Build paths in the saved order rather than sorting or reshuffling the lists.
    fitting_paths = [source_folder / name for name in training]
    validation_paths = [source_folder / name for name in validation]
    checks["filename_order_preserved"] = (
        [path.name for path in fitting_paths] == training
        and [path.name for path in validation_paths] == validation
    )
    # Record the source manifest's fingerprint so later runs can identify this exact split.
    report = {
        "source_manifest": str(split_file.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_split_seed": split["seed"],
        "source_folder": str(source_folder.resolve()),
        "fitting_images": len(fitting_paths),
        "validation_images": len(validation_paths),
        "overlap_count": len(set(training) & set(validation)),
        "checks": checks,
    }
    return fitting_paths, validation_paths, report


# Collect the chosen image/latent dimensions and shared training settings for reporting.
def training_configuration(experiment, seed=42, batch_size=8, learning_rate=0.001,
                           epochs=50, patience=10):
    """Describe the shared training settings and the selected image/latent dimensions."""
    if experiment not in EXPERIMENTS:
        raise ValueError("Choose the E1, E2 or E3 configuration.")
    resolution, latent_dimension = EXPERIMENTS[experiment]
    return {
        "experiment": experiment,
        "input_resolution": resolution,
        "latent_dimension": latent_dimension,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "maximum_epochs": epochs,
        "patience": patience,
        "optimizer": "Adam",
        "device": "cpu",
        "cpu_threads": 4,
        "data_loader_workers": 0,
        "torch_version": str(torch.__version__),
        "loss": "half summed squared RGB error + summed latent KL; batch means",
        "reporting_only": "reconstruction_mse = 2*R/(3*H*W); kl_per_dimension = K/latent_dimension",
        "checkpoint_selection": "strictly lowest validation total R+K; patience counts epochs without improvement",
        "validation_sampling": "one sample per image; seed+1 reset each epoch; no weight updates",
    }


# Make one pass through the images: update weights for fitting, or measure validation loss.
def run_epoch(model, loader, optimizer=None):
    """Fit when an optimizer is supplied; otherwise measure loss without updates."""
    fitting = optimizer is not None
    # Set the model mode; weights change only when the optimizer is used below.
    model.train(fitting)
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    images_seen, optimizer_steps = 0, 0
    # Calculate gradients only for fitting; validation does not need them.
    with torch.set_grad_enabled(fitting):
        for images in loader:
            if fitting:
                # Clear the previous batch's gradients before calculating new ones.
                optimizer.zero_grad(set_to_none=True)
            # Validation also samples: its loss estimates the same VAE objective.
            reconstruction, mean, log_variance = model(images, sample=True)
            # The loss function returns total (R + K), reconstruction R and latent KL K.
            losses = vae_loss(reconstruction, images, mean, log_variance)
            # These normalized values are for reporting only; gradients use R+K.
            reconstruction_mse = 2 * losses[1] / images[0].numel()
            kl_per_dimension = losses[2] / mean.shape[1]
            metrics = (*losses, reconstruction_mse, kl_per_dimension)
            if not all(bool(torch.isfinite(value)) for value in metrics):
                raise RuntimeError("Non-finite loss; training stopped to avoid an invalid checkpoint.")
            if fitting:
                # Work out how each model parameter affects the total loss.
                losses[0].backward()
                # Adam uses those gradients to adjust the weights.
                optimizer.step()
                optimizer_steps += 1
            batch_count = images.shape[0]
            # Weight batch means by their image counts, including the final small batch.
            for index, loss in enumerate(metrics):
                totals[index] += float(loss.detach()) * batch_count
            images_seen += batch_count
    # Divide accumulated totals by image count to report mean losses for the whole epoch.
    return {
        "total": totals[0] / images_seen,
        "reconstruction": totals[1] / images_seen,
        "kl": totals[2] / images_seen,
        "reconstruction_mse": totals[3] / images_seen,
        "kl_per_dimension": totals[4] / images_seen,
        "images": images_seen,
        "optimizer_steps": optimizer_steps,
    }


# Train a new model and save its split, history, best checkpoint and run summary.
def train_vae(args):
    """Save the split and loss history, and select a checkpoint using normal validation."""
    # Fitting and validation use normal training images only, never reserved test images.
    source_folder = args.data_root / "train" / "good"
    configuration = training_configuration(args.experiment, seed=args.seed, batch_size=args.batch_size,
                                           learning_rate=args.learning_rate, epochs=args.epochs,
                                           patience=args.patience)
    resolution, latent_dimension = configuration["input_resolution"], configuration["latent_dimension"]
    split_file = args.split_file
    # E2/E3 default to E1's saved split; E1 creates a split if none is supplied.
    if args.experiment in ("E2", "E3") and split_file is None:
        split_file = E1_SPLIT_FILE
    split_verification = None
    if split_file is not None:
        fitting_paths, validation_paths, split_verification = load_saved_split(source_folder, split_file)
        if args.seed != split_verification["source_split_seed"]:
            raise ValueError("Use the saved E1 split's seed to keep the agreed training seeds consistent.")
    else:
        fitting_paths, validation_paths = split_normal_images(source_folder, args.seed)
    # Use the requested run name or generate one from the current UTC time.
    run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                           for character in run_name):
        raise ValueError("Run name must contain only letters, digits, underscores or hyphens.")
    output_folder = Path("outputs") / args.experiment / run_name
    # A new folder protects all earlier results from accidental overwrite.
    output_folder.mkdir(parents=True, exist_ok=False)
    # Fix model initialization and fitting-order seeds, using the agreed four CPU threads.
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(args.seed)
    # Shuffle fitting images each epoch; keep validation in the same saved order.
    fitting_loader = DataLoader(NormalBottleDataset(fitting_paths, resolution), batch_size=args.batch_size,
                                shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(NormalBottleDataset(validation_paths, resolution), batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)
    # Start with fresh weights at the selected resolution and latent size, then use Adam.
    model = BottleVAE(image_size=resolution, latent_dim=latent_dimension).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Save filename lists only; no original images are copied or modified.
    split = {
        "source_folder": str(source_folder.resolve()),
        "seed": args.seed,
        "training": [path.name for path in fitting_paths],
        "validation": [path.name for path in validation_paths],
    }
    if split_file is None:
        (output_folder / "data_split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    else:
        # Copy the manifest bytes, not dataset images; this also retains the source hash.
        (output_folder / "data_split.json").write_bytes(split_file.read_bytes())
    # Record settings and actual progress so the completed run can be explained later.
    summary = {
        "stage": f"{args.experiment} training; defect-detection evaluation has not been performed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_verification": split_verification,
        "training_images": len(fitting_paths),
        "validation_images": len(validation_paths),
        "epochs_completed": 0,
        "optimizer_steps": 0,
        "best_epoch": None,
        "best_validation_loss": None,
        "checkpoint": None,
        "status": "running",
    }
    history_path = output_folder / "history.csv"
    metric_names = ("total", "reconstruction", "kl", "reconstruction_mse", "kl_per_dimension")
    fieldnames = ["epoch", *(f"train_{name}" for name in metric_names),
                  *(f"validation_{name}" for name in metric_names), "seconds"]
    # Infinity lets the first finite validation result become the initial best checkpoint.
    best_loss, stale_epochs = float("inf"), 0
    # Time the training loop, including its history and checkpoint writes.
    started = time.perf_counter()
    print(f"{args.experiment}: {len(fitting_paths)} fitting images, {len(validation_paths)} validation images", flush=True)
    print(f"Outputs: {output_folder.resolve()}", flush=True)
    try:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                epoch_started = time.perf_counter()
                # First learn from fitting images, then measure held-out normal validation.
                fitting = run_epoch(model, fitting_loader, optimizer)
                # Reuse validation noise each epoch and restore the fitting RNG afterward.
                with torch.random.fork_rng(devices=[]):
                    # The same validation seed gives comparable latent samples each epoch.
                    torch.manual_seed(args.seed + 1)
                    validation = run_epoch(model, validation_loader)
                seconds = time.perf_counter() - epoch_started
                row = {"epoch": epoch, "seconds": seconds}
                row.update({f"train_{name}": fitting[name] for name in metric_names})
                row.update({f"validation_{name}": validation[name]
                            for name in metric_names})
                # Write and flush each completed epoch so the saved history stays up to date.
                writer.writerow(row)
                stream.flush()
                summary["epochs_completed"] = epoch
                summary["optimizer_steps"] += fitting["optimizer_steps"]
                # Select weights using strictly lower validation R + K, not fitting loss.
                if validation["total"] < best_loss:
                    best_loss, stale_epochs = validation["total"], 0
                    checkpoint = output_folder / "best_model.pt"
                    # Replace this run's best checkpoint only when validation improves.
                    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                                "validation_loss": best_loss, "configuration": configuration}, checkpoint)
                    summary.update(best_epoch=epoch, best_validation_loss=best_loss,
                                   checkpoint=checkpoint.name)
                else:
                    # Equal or higher validation loss counts as an epoch without improvement.
                    stale_epochs += 1
                print(f"Epoch {epoch:02d}/{args.epochs}: train={fitting['total']:.3f}, "
                      f"validation={validation['total']:.3f}, "
                      f"validation reconstruction={validation['reconstruction']:.3f}, "
                      f"validation KL={validation['kl']:.3f}, {seconds:.1f}s", flush=True)
                # Stop after the requested number of consecutive epochs without improvement.
                if stale_epochs >= args.patience:
                    summary["status"] = "early_stopped"
                    break
            else:
                # A for-loop's else runs when it finishes normally, without an early break.
                summary["status"] = "completed_requested_epochs"
    except Exception as error:
        # Record the failure and stop; do not silently change settings or continue training.
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Save timing and status whether the training loop completes, stops early or fails.
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = time.perf_counter() - started
        (output_folder / "training_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(f"Finished: {summary['status']}; best epoch {summary['best_epoch']}", flush=True)
    return output_folder


# Read command-line training options, check their values and start the requested run.
def main(argv=None):
    """Read training settings and start a new run on the actual normal dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="E1")
    parser.add_argument("--data-root", type=Path, default=Path("data/bottle"))
    parser.add_argument("--split-file", type=Path,
                        help="Saved filename split; E2/E3 default to outputs/E1/full_training/data_split.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", help="Optional new output folder name; existing folders are refused")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("Epochs, batch size and patience must be positive; learning rate must be finite and positive.")
    train_vae(args)
    return 0


# Start the command-line workflow only when this file is run directly.
if __name__ == "__main__":
    raise SystemExit(main())
