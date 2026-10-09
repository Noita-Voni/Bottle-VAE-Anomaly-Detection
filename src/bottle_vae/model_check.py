"""Check an untrained VAE using real normal training images on the CPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__:
    from .loss import vae_loss
    from .model import BottleVAE, E1VAE
    from .preprocessing import EXPERIMENTS
    from .train import E1_SPLIT_FILE, NormalBottleDataset, load_saved_split, run_epoch, training_configuration
else:  # Also allow running this file directly from PowerShell.
    from loss import vae_loss
    from model import BottleVAE, E1VAE
    from preprocessing import EXPERIMENTS
    from train import E1_SPLIT_FILE, NormalBottleDataset, load_saved_split, run_epoch, training_configuration


# Prepare three actual normal training images for checking the model's dimensions.
def load_training_batch(source_folder, resolution=128):
    """Load first, middle and last actual PNGs through the configurable dataset loader."""
    paths = sorted(path for path in source_folder.iterdir()
                   if path.is_file() and path.suffix.lower() == ".png")
    if len(paths) < 3:
        raise ValueError(f"Need at least three real training PNGs in {source_folder}.")
    # Select the first, middle and last sorted filenames without creating image data.
    selected = [paths[0], paths[(len(paths) - 1) // 2], paths[-1]]
    dataset = NormalBottleDataset(selected, resolution=resolution)
    images = [dataset[index] for index in range(len(dataset))]
    # Stack the three CHW images into one NCHW batch: images, channels, height, width.
    return selected, torch.stack(images)


# Summarise an array of values without saving the full array or image.
def tensor_summary(tensor):
    """Record shape, type, range and finiteness without saving image arrays."""
    # Finite means every value is a number other than NaN or positive/negative infinity.
    finite = bool(torch.isfinite(tensor).all())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "all_finite": finite,
        "value_range": [float(tensor.min()), float(tensor.max())] if finite else None,
    }


# Confirm that the reported losses match direct calculations on the same real images.
def check_reporting_metrics(model, batch):
    """Compare logged averages with direct calculations; no optimizer or gradients."""
    # Two plus one real images also check weighting of the final smaller batch.
    loader = DataLoader(batch, batch_size=2, shuffle=False, num_workers=0)
    reference = {"total": 0.0, "reconstruction_mse": 0.0, "kl_per_dimension": 0.0}
    # Restore the original random state afterward so this check does not alter it.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        # No optimizer is supplied: measure losses without updating model weights.
        measured = run_epoch(model, loader)
        # Reset the seed so the direct calculation uses the same sampled latent values.
        torch.manual_seed(43)
        with torch.no_grad():
            for images in loader:
                reconstruction, mean, log_variance = model(images, sample=True)
                squared = (reconstruction - images).square()
                latent_kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp())
                # Keep R + K as the objective; MSE and KL per dimension are reporting only.
                direct = {
                    "total": 0.5 * squared.flatten(1).sum(1).mean() + latent_kl.sum(1).mean(),
                    "reconstruction_mse": squared.mean(),
                    "kl_per_dimension": latent_kl.mean(),
                }
                for name, value in direct.items():
                    # Weight by image count so the smaller final batch is counted correctly.
                    reference[name] += float(value) * images.shape[0] / len(batch)
    # Allow small floating-point rounding differences when comparing the calculations.
    checks = {f"{name}_matches_direct_calculation": math.isclose(
        measured[name], expected, rel_tol=1e-6, abs_tol=1e-8
    ) for name, expected in reference.items()}
    checks["zero_optimizer_steps"] = measured["optimizer_steps"] == 0
    return {"measured": measured, "direct_reference": reference,
            "relative_tolerance": 1e-6, "absolute_tolerance": 1e-8, "checks": checks}


# Confirm that existing E1/E2 checkpoints still load with the configurable model code.
def check_checkpoint_compatibility(selected, configuration, experiment="E1"):
    """Strictly load saved weights and reconstruct real normal images without updates."""
    checkpoint_path = Path("outputs") / experiment / "full_training" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    resolution, latent_dimension = EXPERIMENTS[experiment]
    model = (E1VAE() if experiment == "E1" else
             BottleVAE(image_size=resolution, latent_dim=latent_dimension)).cpu()
    # Strict loading fails if saved parameter names or dimensions no longer match.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = (NormalBottleDataset(selected) if experiment == "E1" else
               NormalBottleDataset(selected, resolution=resolution))
    batch = torch.stack([dataset[index] for index in range(len(dataset))])
    with torch.inference_mode():
        reconstruction, mean, log_variance = model(batch)
    # Compare common training settings; each experiment retains its own image/latent size.
    shared_settings = ("seed", "batch_size", "learning_rate", "maximum_epochs", "patience",
                       "optimizer", "device", "cpu_threads", "torch_version", "loss", "validation_sampling")
    checks = {
        "strict_checkpoint_load": True,  # load_state_dict would raise on incompatible keys/shapes.
        "original_dimensions": model.image_size == resolution and model.latent_dim == latent_dimension,
        "original_parameter_count": sum(p.numel() for p in model.parameters())
                                    == {"E1": 747395, "E2": 3524515}[experiment],
        "input_output_shape": tuple(batch.shape) == (3, 3, resolution, resolution) == tuple(reconstruction.shape),
        "latent_shape": tuple(mean.shape) == (3, latent_dimension) == tuple(log_variance.shape),
        "finite_reconstruction": bool(torch.isfinite(reconstruction).all()),
        "shared_training_settings_match": all(
            configuration[name] == checkpoint["configuration"][name] for name in shared_settings
        ),
    }
    return {"checkpoint": str(checkpoint_path.resolve()), "epoch": checkpoint["epoch"],
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "compared_training_settings": list(shared_settings), "checks": checks}


# Check the requested batch size using actual fitting images, without training.
def check_full_batch(paths, configuration):
    """Verify the configured batch using actual fitting images; no backward pass or updates."""
    batch_size = configuration["batch_size"]
    resolution, latent_dimension = configuration["input_resolution"], configuration["latent_dimension"]
    # Preserve the saved fitting order and use the first full batch of filenames.
    selected = paths[:batch_size]
    loader = DataLoader(NormalBottleDataset(selected, resolution=resolution),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model = BottleVAE(image_size=resolution, latent_dim=latent_dimension).cpu()
    model.eval()
    # Run reconstruction and R + K only; backward-pass and optimizer memory are not tested.
    with torch.inference_mode():
        reconstruction, mean, log_variance = model(batch, sample=True)
        total, reconstruction_loss, kl = vae_loss(reconstruction, batch, mean, log_variance)
    if not all(bool(torch.isfinite(value)) for value in (total, reconstruction_loss, kl)):
        raise RuntimeError(f"Batch size {batch_size} produced non-finite losses.")
    # Check batch size, image/latent dimensions, finite values and absence of gradients.
    checks = {
        "requested_batch_size_loaded": len(batch) == batch_size,
        "input_output_shape": tuple(batch.shape) == (batch_size, 3, resolution, resolution)
                              == tuple(reconstruction.shape),
        "latent_shapes": tuple(mean.shape) == (batch_size, latent_dimension) == tuple(log_variance.shape),
        "all_finite": all(bool(torch.isfinite(value).all())
                          for value in (batch, reconstruction, mean, log_variance)),
        "no_gradients_created": all(parameter.grad is None for parameter in model.parameters()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "requested_batch_size": batch_size,
        "source_files": [str(path.resolve()) for path in selected],
        "selection": "first fitting filenames in the unchanged saved manifest order",
        "input": tensor_summary(batch), "reconstruction": tensor_summary(reconstruction),
        "mean": tensor_summary(mean), "log_variance": tensor_summary(log_variance),
        "losses": {"total": float(total), "reconstruction": float(reconstruction_loss),
                   "kl": float(kl), "reconstruction_mse": float(2 * reconstruction_loss / batch[0].numel()),
                   "kl_per_dimension": float(kl / latent_dimension)},
        "optimizer_steps": 0,
        "scope": "forward pass and R+K only; backward-pass and Adam-state memory have not been checked",
        "checks": checks,
    }


# Check shapes, value ranges and latent sampling in a newly initialised, untrained model.
def check_model(batch, resolution=128, latent_dimension=16):
    """Verify forward behaviour with fixed initialization and no weight updates."""
    model = BottleVAE(image_size=resolution, latent_dim=latent_dimension).cpu()
    # Copy the initial weights so we can confirm that none change during these checks.
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.eval()
    with torch.inference_mode():
        # In evaluation mode, the default reconstruction uses the latent mean.
        reconstruction, mean, log_variance = model(batch)
        repeated, _, _ = model(batch)
        single, _, _ = model(batch[:1])
        features = model.encoder(batch)
        # Random latent sampling should vary, while mean-only reconstruction repeats.
        latent = model.reparameterize(mean, log_variance)
        another_latent = model.reparameterize(mean, log_variance)
        sampled, _, _ = model(batch, sample=True)
        model.train()  # Checks the sampling mode only; no learning takes place.
        training_mode_output, _, _ = model(batch)
        mean_override, _, _ = model(batch, sample=False)

    # Record summaries of the inputs, encoded values and reconstructions for the report.
    tensors = {
        "input": batch,
        "encoder_features": features,
        "mean": mean,
        "log_variance": log_variance,
        "sampled_latent": latent,
        "another_sampled_latent": another_latent,
        "reconstruction": reconstruction,
        "sampled_reconstruction": sampled,
        "training_mode_reconstruction": training_mode_output,
        "single_image_reconstruction": single,
    }
    summaries = {name: tensor_summary(value) for name, value in tensors.items()}
    reporting = check_reporting_metrics(model, batch)
    image_outputs = (reconstruction, repeated, sampled, training_mode_output, mean_override, single)
    # Check expected shapes, float32 values, valid ranges and unchanged model weights.
    checks = {
        "input_shape": tuple(batch.shape) == (3, 3, resolution, resolution),
        "input_range": bool(((batch >= 0) & (batch <= 1)).all()),
        "encoder_feature_shape": tuple(features.shape) == (3, 128, resolution // 16, resolution // 16),
        "latent_shapes": all(tuple(value.shape) == (3, latent_dimension)
                             for value in (mean, log_variance, latent, another_latent)),
        "reconstruction_shapes": all(value.shape == batch.shape for value in image_outputs[:-1]),
        "single_image_shape": tuple(single.shape) == (1, 3, resolution, resolution),
        "all_float32": all(value.dtype == torch.float32 for value in tensors.values()),
        "all_finite": all(bool(torch.isfinite(value).all()) for value in tensors.values()),
        "output_range": all(bool(((value >= 0) & (value <= 1)).all()) for value in image_outputs),
        "evaluation_repeatable": torch.equal(reconstruction, repeated),
        "explicit_mean_override": torch.equal(reconstruction, mean_override),
        "latent_sampling_varies": not torch.equal(latent, another_latent),
        "parameters_unchanged": all(torch.equal(before[name], value)
                                    for name, value in model.state_dict().items()),
    }
    checks.update(reporting["checks"])
    # Count all model parameters and show how they are distributed across its layers.
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "flattened_features_per_image": features[0].numel(),
        "parameter_groups": {name: sum(parameter.numel() for parameter in module.parameters())
                             for name, module in (("encoder_convolutions", model.encoder),
                                                  ("mean_head", model.mean_head),
                                                  ("log_variance_head", model.log_variance_head),
                                                  ("decoder_input", model.decoder_input),
                                                  ("decoder_convolutions", model.decoder))},
        "tensors": summaries,
        "reporting_verification": reporting,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


# Read the selected experiment, run its preparation checks and save their results.
def main(argv=None):
    """Check real-image forward passes and save a clearly labelled architecture report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="E1")
    parser.add_argument("--data-root", type=Path, default=Path("data/bottle"))
    parser.add_argument("--split-file", type=Path, default=E1_SPLIT_FILE,
                        help="E1 saved manifest to check for E2/E3; never reshuffle its filename lists")
    args = parser.parse_args(argv)
    # Use only actual normal training images; no reserved test images or masks are read.
    source_folder = args.data_root / "train" / "good"
    configuration = training_configuration(args.experiment)
    resolution, latent_dimension = configuration["input_resolution"], configuration["latent_dimension"]
    # Use the agreed seed and four CPU threads for the preparation checks.
    torch.manual_seed(42)
    torch.set_num_threads(4)
    report = {
        "stage": f"Untrained {args.experiment} preparation/architecture check; no performance evaluation",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(source_folder.resolve()),
        "source_files": [],
        "selection": "first, middle and last sorted training PNG filenames",
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": 42,
        "cpu_threads": 4,
        "input_resolution": resolution,
        "latent_dimension": latent_dimension,
        "proposed_training_configuration": configuration,
        "layout": "NCHW: batch, channels, height, width",
        "trained": False,
        "optimizer_steps": 0,
        "failures": [],
    }
    try:
        # E2/E3 must reuse E1's saved fitting/validation membership and filename order.
        if args.experiment in ("E2", "E3"):
            fitting, validation, split_report = load_saved_split(source_folder, args.split_file)
            if split_report["source_split_seed"] != configuration["seed"]:
                raise ValueError("Saved E1 split seed differs from the proposed training seed.")
            saved_split = json.loads(args.split_file.read_text(encoding="utf-8"))
            # Dataset lists retain the manifest order; DataLoader will shuffle fitting only at training time.
            split_report["checks"]["dataset_filename_order_preserved"] = (
                [path.name for path in NormalBottleDataset(fitting, resolution).paths]
                == saved_split["training"]
                and [path.name for path in NormalBottleDataset(validation, resolution).paths]
                == saved_split["validation"]
            )
            report["split_verification"] = split_report
            for name, passed in split_report["checks"].items():
                if not passed:
                    report["failures"].append(f"split: {name}")
        # Run the three-image architecture check and collect any failures.
        selected, batch = load_training_batch(source_folder, resolution)
        report["source_files"] = [str(path.resolve()) for path in selected]
        model_report = check_model(batch, resolution, latent_dimension)
        report["failures"].extend(model_report.pop("failures"))
        report.update(model_report)
        # Compare the implemented parameter count with the verified E2/E3 architecture.
        if args.experiment in ("E2", "E3"):
            parameter_check = f"expected_{args.experiment.lower()}_parameter_count"
            report["checks"][parameter_check] = report["parameter_count"] == {
                "E2": 3524515, "E3": 25642979
            }[args.experiment]
            if not report["checks"][parameter_check]:
                report["failures"].append(parameter_check)
            # E3 also checks its requested batch size; a failure does not change settings.
            if args.experiment == "E3":
                report["batch_size_check"] = {"requested_batch_size": configuration["batch_size"], "status": "FAIL"}
                report["batch_size_check"].update(check_full_batch(fitting, configuration))
                for name, passed in report["batch_size_check"]["checks"].items():
                    if not passed:
                        report["failures"].append(f"batch size 8: {name}")
            # Check that adding this configuration still allows earlier checkpoints to load.
            for experiment in (("E1", "E2") if args.experiment == "E3" else ("E1",)):
                compatibility = check_checkpoint_compatibility(selected, configuration, experiment)
                report[f"{experiment.lower()}_compatibility"] = compatibility
                for name, passed in compatibility["checks"].items():
                    if not passed:
                        report["failures"].append(f"{experiment} compatibility: {name}")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, MemoryError) as error:
        # Record the error so the final report explains why a check failed.
        report["failures"].append(f"{type(error).__name__}: {error}")
    output = Path("outputs") / args.experiment / "model_check_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Save E2/E3 configuration and split records only after the checks have passed.
    if args.experiment in ("E2", "E3") and not report["failures"]:
        (output.parent / "training_configuration.json").write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        # Copy only the filename manifest, not images; its fingerprint must stay identical.
        (output.parent / "data_split.json").write_bytes(args.split_file.read_bytes())
        split_report["copied_manifest_sha256"] = hashlib.sha256(
            (output.parent / "data_split.json").read_bytes()
        ).hexdigest()
        split_report["checks"]["copied_manifest_hash_matches_source"] = (
            split_report["copied_manifest_sha256"] == split_report["source_manifest_sha256"]
        )
        if not split_report["checks"]["copied_manifest_hash_matches_source"]:
            report["failures"].append("copied_manifest_hash_matches_source")
        (output.parent / "split_verification.json").write_text(
            json.dumps(report["split_verification"], indent=2) + "\n", encoding="utf-8"
        )
    # Save the preparation report and print its results; no trained checkpoint is created.
    report["status"] = "FAIL" if report["failures"] else "PASS"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report["stage"])
    for path in report["source_files"]:
        print(f"Source: {path}")
    for name, passed in report.get("checks", {}).items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print(f"Parameters: {report.get('parameter_count')}")
    if "batch_size_check" in report:
        batch_report = report["batch_size_check"]
        print(f"Batch size {batch_report['requested_batch_size']} forward/loss check: {batch_report['status']}; no weight updates")
    if "split_verification" in report:
        split_report = report["split_verification"]
        print(f"Split: {split_report['fitting_images']} fitting / {split_report['validation_images']} validation; "
              f"overlap: {split_report['overlap_count']}; filename order preserved")
        print(f"Source manifest SHA256: {split_report['source_manifest_sha256']}")
    for failure in report["failures"]:
        print(f"  - {failure}")
    print(f"{report['status']}: {output}")
    # Exit code 0 means success; 1 tells the command line that a check failed.
    return 1 if report["failures"] else 0


# Start the command-line workflow only when this file is run directly.
if __name__ == "__main__":
    raise SystemExit(main())
