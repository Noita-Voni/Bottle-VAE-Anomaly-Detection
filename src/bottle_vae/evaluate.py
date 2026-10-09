"""Evaluate frozen E1/E2/E3 models on the same actual reserved bottle images."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")  # Save only the requested ROC figures, without desktop windows.
import matplotlib.pyplot as plt
from PIL import Image

if __package__:
    from .scoring import load_model, score_images
else:
    from scoring import load_model, score_images


# Benchmark counts are compared with actual files; they do not replace actual counts.
EXPECTED_COUNTS = {"good": 20, "broken_large": 20, "broken_small": 22, "contamination": 21}
# Each tuple is (image side length, latent dimensions, selected checkpoint epoch).
FROZEN_RUNS = {"E1": (128, 16, 47), "E2": (256, 32, 45), "E3": (512, 64, 13)}


# Create a fingerprint of a file's contents to identify the exact saved file.
def file_hash(path):
    """Identify an unchanged source or saved artifact by its SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Read the reserved test images and record their counts, labels and readability.
def inspect_test_images(data_root):
    """Count and fully decode every required test PNG; record all failures."""
    test_folder = data_root / "test"
    report = {"source_folder": str(test_folder.resolve()), "categories": {},
              "images": [], "discrepancies": [], "failures": []}
    paths = []
    if not test_folder.is_dir():
        report["failures"].append(f"Missing test folder: {test_folder}")
    else:
        for entry in sorted(test_folder.iterdir()):
            if entry.name not in EXPECTED_COUNTS or not entry.is_dir():
                report["failures"].append(f"Unexpected test entry: {entry}")
    for category, expected in EXPECTED_COUNTS.items():
        folder = test_folder / category
        entries = sorted(folder.iterdir()) if folder.is_dir() else []
        files = [path for path in entries if path.is_file()]
        readable = 0
        if not folder.is_dir():
            report["failures"].append(f"Missing required folder: {folder}")
        if len(files) != expected:
            report["discrepancies"].append(f"{category}: expected {expected} files; found {len(files)}.")
        for path in entries:
            if not path.is_file() or path.suffix.lower() != ".png":
                report["failures"].append(f"Unexpected non-PNG entry: {path}")
                continue
            try:
                # Check the PNG structure, then reopen it to fully read its pixels.
                with Image.open(path) as original:
                    if original.format != "PNG":
                        raise ValueError("File contents are not PNG.")
                    original.verify()
                with Image.open(path) as original:
                    original.load()
                    size, mode = list(original.size), original.mode
                # Folder names supply the true labels: good = 0, defective = 1.
                report["images"].append({
                    "source_path": path.resolve().relative_to(Path.cwd()).as_posix(),
                    "category": category, "actual_label": int(category != "good"),
                    "sha256": file_hash(path), "size_wh": size, "mode": mode,
                })
                paths.append(path)
                readable += 1
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                report["failures"].append(f"{path}: {type(error).__name__}: {error}")
        report["categories"][category] = {
            "expected": expected, "actual_files": len(files), "readable_images": readable,
        }
        print(f"{category}: expected {expected}, found {len(files)}, readable {readable}", flush=True)
    report["total_images_read"] = len(paths)
    report["all_images_readable"] = not report["failures"]
    return paths, report


# Load the selected model and its existing threshold without training or recalibration.
def load_frozen_run(experiment, run_folder):
    """Reuse selected weights and verify the frozen calibration's provenance."""
    threshold_path = run_folder / "calibration" / "threshold.json"
    calibration = json.loads(threshold_path.read_text(encoding="utf-8"))
    checkpoint_path, split_path = run_folder / "best_model.pt", run_folder / "data_split.json"
    checkpoint_hash, split_hash = file_hash(checkpoint_path), file_hash(split_path)
    model, checkpoint, summary = load_model(run_folder)
    resolution, latent_dimension, epoch = FROZEN_RUNS[experiment]
    configuration = checkpoint["configuration"]
    threshold = calibration["threshold"]
    # Check that dimensions, epoch, files and calibration agree with the saved run.
    checks = {
        "frozen_model_configuration": (configuration["experiment"], model.image_size,
                                        model.latent_dim, checkpoint["epoch"])
                                       == (experiment, resolution, latent_dimension, epoch),
        "calibration_configuration": (calibration["experiment"], calibration["input_resolution"],
                                      calibration["latent_dimension"], calibration["checkpoint_epoch"])
                                     == (experiment, resolution, latent_dimension, epoch),
        "checkpoint_path": Path(calibration["checkpoint"]).resolve() == checkpoint_path.resolve(),
        "checkpoint_hash": calibration["checkpoint_sha256"] == checkpoint_hash,
        "split_path": Path(calibration["data_split"]).resolve() == split_path.resolve(),
        "split_hash": calibration["data_split_sha256"] == split_hash,
        "threshold_finite_nonnegative": isinstance(threshold, (int, float))
                                        and math.isfinite(threshold) and threshold >= 0,
        "recorded_percentile_method": calibration["percentile"] == 95
                                      and calibration["percentile_method"] == "numpy.percentile, method=linear",
        "recorded_decision_rule": calibration["decision_rule"]
                                  == "flag as anomaly when score > threshold; equality stays normal",
        "recorded_reconstruction": calibration["reconstruction"]
                                   == "latent mean; sample=False; no weight updates",
        "selected_loss_matches_summary": checkpoint["validation_loss"] == summary["best_validation_loss"],
        "summary_configuration": summary["configuration"] == configuration,
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if "parameter_count" in summary:
        checks["parameter_count_matches_training"] = parameter_count == summary["parameter_count"]
    if not all(checks.values()):
        raise ValueError(f"{experiment} frozen provenance failed: "
                         + ", ".join(name for name, passed in checks.items() if not passed))
    # Record which files and settings produced this evaluation so it can be traced.
    provenance = {
        "experiment": experiment, "input_resolution": resolution, "latent_dimension": latent_dimension,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": epoch,
        "checkpoint_sha256": checkpoint_hash, "calibration": str(threshold_path.resolve()),
        "calibration_sha256": file_hash(threshold_path), "threshold": threshold,
        "data_split_sha256": split_hash, "frozen_checks": checks,
        "training_summary": str((run_folder / "training_summary.json").resolve()),
        "training_summary_sha256": file_hash(run_folder / "training_summary.json"),
        "training_seconds": summary["elapsed_seconds"], "training_epochs": summary["epochs_completed"],
        "training_status": summary["status"], "parameter_count": parameter_count,
    }
    return model, threshold, provenance


# A zero denominator makes a metric undefined; None is saved as JSON null.
def safe_ratio(numerator, denominator):
    """Return null for an undefined ratio rather than silently substituting zero."""
    return numerator / denominator if denominator else None


# Calculate ROC and AUC from the scores, rather than the final yes/no predictions.
def calculate_roc(rows):
    """Sweep complete equal-score groups; integrate ROC and check pairwise tie credit."""
    positive = [row["score"] for row in rows if row["actual_label"] == 1]
    negative = [row["score"] for row in rows if row["actual_label"] == 0]
    tie_groups = sum(count > 1 for count in Counter(row["score"] for row in rows).values())
    if not positive or not negative:
        return [], None, {"tied_score_groups": tie_groups, "pairwise_auc": None,
                          "pairwise_auc_matches": None, "reason": "ROC-AUC requires both actual classes."}
    # Start with no images flagged, then include images from highest to lowest score.
    points = [{"false_positive_rate": 0.0, "true_positive_rate": 0.0, "TP": 0, "FP": 0}]
    tp, fp = 0, 0
    for score, group in groupby(sorted(rows, key=lambda row: row["score"], reverse=True),
                                key=lambda row: row["score"]):
        # Equal scores cross the boundary together, independent of filename order.
        for row in group:
            tp += row["actual_label"] == 1
            fp += row["actual_label"] == 0
        points.append({"false_positive_rate": fp / len(negative),
                       "true_positive_rate": tp / len(positive), "TP": tp, "FP": fp})
    # Add the trapezoid areas between neighbouring points to get the area under ROC.
    auc = sum((right["false_positive_rate"] - left["false_positive_rate"])
              * (right["true_positive_rate"] + left["true_positive_rate"]) / 2
              for left, right in zip(points, points[1:]))
    # Only 63 * 20 pairs here; this independent check awards half credit to ties.
    # AUC also measures how often a defect's score exceeds a normal image's score.
    pairwise_auc = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    pairwise_auc /= len(positive) * len(negative)
    verification = {"tied_score_groups": tie_groups, "pairwise_auc": pairwise_auc,
                    "pairwise_auc_matches": math.isclose(auc, pairwise_auc, rel_tol=1e-12, abs_tol=1e-12),
                    "endpoints_correct": points[0]["TP"] == points[0]["FP"] == 0
                                         and points[-1]["TP"] == len(positive)
                                         and points[-1]["FP"] == len(negative)}
    if not verification["pairwise_auc_matches"] or not verification["endpoints_correct"]:
        raise ValueError("ROC verification failed.")
    return points, auc, verification


# Compare predicted labels with true labels and summarise detection performance.
def calculate_metrics(rows, actual_counts):
    """Count image decisions and reconcile positive/normal and category totals."""
    # Defective is positive (1): TP = detected defect, FN = missed defect.
    # Normal is negative (0): FP = normal flagged defective, TN = normal accepted.
    tp = sum(row["actual_label"] == 1 and row["predicted_label"] == 1 for row in rows)
    fn = sum(row["actual_label"] == 1 and row["predicted_label"] == 0 for row in rows)
    fp = sum(row["actual_label"] == 0 and row["predicted_label"] == 1 for row in rows)
    tn = sum(row["actual_label"] == 0 and row["predicted_label"] == 0 for row in rows)
    # Also report detected and missed defects separately for each defect category.
    categories = {}
    for category in EXPECTED_COUNTS:
        members = [row for row in rows if row["category"] == category]
        flagged = sum(row["predicted_label"] == 1 for row in members)
        if category == "good":
            categories[category] = {"images": len(members), "FP": flagged, "TN": len(members) - flagged}
        else:
            categories[category] = {"images": len(members), "detected": flagged,
                                    "missed": len(members) - flagged,
                                    "missed_source_paths": [row["source_path"] for row in members
                                                            if row["predicted_label"] == 0]}
    points, auc, roc_verification = calculate_roc(rows)
    # Precision: correct defect flags / all flags. Recall: detected / actual defects.
    # F1 combines precision and recall; the normal FP rate counts normal images flagged.
    metrics = {
        "positive_class": "defective (1); normal (0)", "evaluated_images": len(rows),
        "confusion_matrix": {"rows_actual": ["defective", "normal"],
                             "columns_predicted": ["defective", "normal"],
                             "counts": [[tp, fn], [fp, tn]], "TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "precision": safe_ratio(tp, tp + fp), "recall": safe_ratio(tp, tp + fn),
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "normal_false_positive_count": fp, "normal_false_positive_rate": safe_ratio(fp, fp + tn),
        "categories": categories, "roc_auc": auc, "roc_verification": roc_verification,
    }
    metrics["undefined_metrics"] = [name for name in ("precision", "recall", "f1",
                                                      "normal_false_positive_rate", "roc_auc")
                                    if metrics[name] is None]
    metrics["undefined_metric_policy"] = (
        "JSON null / CSV blank when the denominator is zero; ROC-AUC needs both classes. "
        "F1 uses 2TP/(2TP+FP+FN), including zero when actual defects are all missed."
    )
    # Confirm that all totals agree with the actual dataset and no image is counted twice.
    checks = {
        "confusion_total_matches_images": tp + fn + fp + tn == len(rows),
        "normal_total_matches_dataset": fp + tn == actual_counts["good"],
        "defect_total_matches_dataset": tp + fn == sum(actual_counts[name] for name in EXPECTED_COUNTS
                                                       if name != "good"),
        "category_counts_match_dataset": all(categories[name]["images"] == actual_counts[name]
                                              for name in EXPECTED_COUNTS),
        "category_detected_total_matches_TP": sum(values["detected"] for name, values in categories.items()
                                                  if name != "good") == tp,
        "category_missed_total_matches_FN": sum(values["missed"] for name, values in categories.items()
                                                if name != "good") == fn,
        "normal_category_matches_confusion": categories["good"]["FP"] == fp
                                             and categories["good"]["TN"] == tn,
        "unique_source_paths": len({row["source_path"] for row in rows}) == len(rows),
        "all_scores_finite": all(math.isfinite(row["score"]) for row in rows),
    }
    if not all(checks.values()):
        raise ValueError("Evaluation count reconciliation failed.")
    metrics["verification"] = checks
    return metrics, points


# Write a structured report without rounding numbers or allowing NaN/infinity.
def save_json(path, report):
    """Save full-precision numbers and explicit nulls; reject non-finite JSON values."""
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


# Write a table of image decisions, ROC points or experiment comparisons.
def save_csv(path, rows, fieldnames):
    """Save full-precision row values and refuse an existing output file."""
    # Mode "x" creates a new file and fails if that file already exists.
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Save the ROC curve and mark performance at the original calibration threshold.
def save_roc_plot(points, metrics, experiment, output_path):
    """Plot continuous-score ROC and mark the separately frozen decision point."""
    figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    if points:
        axis.plot([point["false_positive_rate"] for point in points],
                  [point["true_positive_rate"] for point in points],
                  label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
        axis.scatter(metrics["normal_false_positive_rate"], metrics["recall"],
                     color="#d55e00", label="Frozen calibration threshold", zorder=3)
    else:
        axis.text(0.5, 0.5, "ROC-AUC undefined: both actual classes are required.",
                  ha="center", va="center", wrap=True, transform=axis.transAxes)
    axis.plot([0, 1], [0, 1], "--", color="#777777", label="Chance diagonal")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate on normal images",
             ylabel="True-positive rate (defect recall)",
             title=f"{experiment}: reserved test image-level ROC\nDefective is positive; continuous RGB MSE scores")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


# Evaluate one configuration using the same test images and its own saved threshold.
def evaluate_configuration(experiment, paths, dataset_report, dataset_text):
    """Score the shared test manifest once using a verified model and saved threshold."""
    run_folder = Path("outputs") / experiment / "full_training"
    output_folder = run_folder / "evaluation"
    output_folder.mkdir(exist_ok=False)
    (output_folder / "dataset_report.json").write_text(dataset_text, encoding="utf-8")
    try:
        model, threshold, provenance = load_frozen_run(experiment, run_folder)
        print(f"{experiment}: verified epoch {provenance['checkpoint_epoch']}; "
              f"unchanged threshold {threshold!r}; scoring {len(paths)} test images", flush=True)
        started = time.perf_counter()
        # Reuse preprocessing and latent-mean RGB MSE scoring with gradients disabled.
        scored = score_images(model, paths, batch_size=8)
        scoring_seconds = time.perf_counter() - started
        if len(scored) != len(paths):
            raise ValueError("Not every test image received a score.")
        rows = []
        # Keep each score paired with the same image and true label in the shared list.
        for path, source, score in zip(paths, dataset_report["images"], scored):
            if score["source_image"] != str(path.resolve()):
                raise ValueError("Scoring order differs from the shared test manifest.")
            # Predict defective only above the saved threshold; equality stays normal.
            rows.append({"source_path": source["source_path"], "category": source["category"],
                         "actual_label": source["actual_label"], "score": score["score"],
                         "predicted_label": int(score["score"] > threshold)})
        counts = {name: values["actual_files"] for name, values in dataset_report["categories"].items()}
        metrics, points = calculate_metrics(rows, counts)
        provenance.update({
            "stage": "Phase 5 image-level reserved test evaluation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_report": str((output_folder / "dataset_report.json").resolve()),
            "dataset_report_sha256": file_hash(output_folder / "dataset_report.json"),
            "score": "mean squared RGB reconstruction error; no KL; existing score_images function",
            "preprocessing": "original converted to RGB; bilinear resize; float32 divided by 255",
            "reconstruction": "latent mean; sample=False; torch.inference_mode; no weight updates",
            "decision_rule": "predicted_label = 1 when score > saved threshold; equality stays normal",
            "label_mapping": {"0": "normal", "1": "defective (positive)"},
            "batch_size": 8, "device": "cpu", "cpu_threads": 4,
            "scoring_seconds": scoring_seconds,
            "roc_method": "descending scores; move equal-score groups together; trapezoidal ROC area",
            "auc_tie_rule": "half credit for equal defect/normal scores in independent pairwise verification",
            "threshold_policy": "loaded at full precision from calibration; never recalculated or tuned",
            "masks_used": False,
        })
        # Save individual decisions, metrics, the ROC plot and a record of their sources.
        save_csv(output_folder / "image_scores.csv", rows,
                 ["source_path", "category", "actual_label", "score", "predicted_label"])
        save_csv(output_folder / "roc_curve.csv", points,
                 ["false_positive_rate", "true_positive_rate", "TP", "FP"])
        save_json(output_folder / "metrics.json", metrics)
        save_json(output_folder / "provenance.json", provenance)
        save_roc_plot(points, metrics, experiment, output_folder / "roc_curve.png")
    except Exception as error:
        # Record the failure and stop instead of skipping images or changing settings.
        save_json(output_folder / "failure_report.json",
                  {"experiment": experiment, "status": "failed", "error": f"{type(error).__name__}: {error}",
                   "policy": "Stopped without skipping images or changing settings; no metrics claimed."})
        raise
    confusion = metrics["confusion_matrix"]
    print(f"{experiment}: TP={confusion['TP']}, FN={confusion['FN']}, "
          f"FP={confusion['FP']}, TN={confusion['TN']}; ROC-AUC={metrics['roc_auc']}", flush=True)
    return {
        "experiment": experiment, "input_resolution": provenance["input_resolution"],
        "latent_dimension": provenance["latent_dimension"], "checkpoint_epoch": provenance["checkpoint_epoch"],
        "threshold": threshold, **{name: confusion[name] for name in ("TP", "FN", "FP", "TN")},
        **{name: metrics[name] for name in ("precision", "recall", "f1", "normal_false_positive_rate", "roc_auc")},
        "training_seconds": provenance["training_seconds"], "training_epochs": provenance["training_epochs"],
        "parameter_count": provenance["parameter_count"],
        "dataset_report_sha256": provenance["dataset_report_sha256"],
        "data_split_sha256": provenance["data_split_sha256"],
    }


# Read command-line options, inspect the test set once and evaluate E1, E2 and E3.
def main(argv=None):
    """Evaluate all frozen configurations; protect all previous results from overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/bottle"))
    args = parser.parse_args(argv)
    if args.data_root.resolve() != Path("data/bottle").resolve():
        parser.error("Use only the actual project dataset at data/bottle.")
    comparison_path = Path("outputs/phase5_comparison.csv")
    output_folders = [Path("outputs") / name / "full_training" / "evaluation" for name in FROZEN_RUNS]
    # Refuse existing evaluation results before starting any image inspection or scoring.
    existing = [str(path) for path in [comparison_path, *output_folders] if path.exists()]
    if existing:
        raise FileExistsError("Evaluation results already exist; refusing to overwrite: " + ", ".join(existing))
    paths, report = inspect_test_images(args.data_root)
    dataset_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    for issue in report["discrepancies"]:
        print(f"Count discrepancy: {issue}", flush=True)
    # Save inspection failures and stop; unreadable or invalid images are not skipped.
    if report["failures"]:
        for folder in output_folders:
            folder.mkdir(exist_ok=False)
            (folder / "dataset_report.json").write_text(dataset_text, encoding="utf-8")
        raise ValueError("Test inspection failed; no images evaluated:\n" + "\n".join(report["failures"]))
    # Pass the same ordered test list to every configuration.
    comparison = [evaluate_configuration(name, paths, report, dataset_text) for name in FROZEN_RUNS]
    # Matching report hashes confirm shared test images and fitting/validation splits.
    if (len({row["dataset_report_sha256"] for row in comparison}) != 1
            or len({row["data_split_sha256"] for row in comparison}) != 1):
        raise ValueError("The configurations did not use the same test manifest and saved normal split.")
    save_csv(comparison_path, comparison, list(comparison[0]))
    print(f"Comparison: {comparison_path.resolve()}", flush=True)
    return 0


# Start the command-line workflow only when this file is run directly.
if __name__ == "__main__":
    raise SystemExit(main())
