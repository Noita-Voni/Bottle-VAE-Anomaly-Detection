"""Check MVTec bottle folders, PNG counts, and image/mask filenames."""

import argparse
from pathlib import Path

DEFECTS = ("broken_large", "broken_small", "contamination")
FOLDERS = ("train/good", "test/good") + tuple(
    f"{split}/{defect}" for split in ("test", "ground_truth") for defect in DEFECTS
)
EXPECTED = {"normal training": 209, "normal test": 20, "defective test": 63}


def check_dataset(root):
    """Return per-folder counts and issues; never change dataset files."""
    root = Path(root)
    counts, images, issues = {}, {}, []
    for relative in FOLDERS:
        folder = root / relative
        counts[relative] = None
        images[relative] = set()
        if not folder.is_dir():
            issues.append(f"Missing folder: {relative}")
            continue
        try:
            entries = list(folder.iterdir())
            pngs = [p for p in entries if p.is_file() and p.suffix.lower() == ".png"]
            counts[relative] = len(pngs)
            images[relative] = {p.stem for p in pngs}
            if not pngs:
                issues.append(f"Empty image folder: {relative}")
            if len(images[relative]) != len(pngs):
                issues.append(f"Duplicate image stems: {relative}")
            for path in entries:
                if path not in pngs:
                    issues.append(f"Unexpected entry: {relative}/{path.name}")
            for path in pngs:
                with path.open("rb") as stream:
                    if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                        issues.append(f"Invalid PNG signature: {relative}/{path.name}")
        except OSError as error:
            issues.append(f"Cannot read {relative}: {error}")

    for split, allowed in (
        ("train", {"good"}),
        ("test", {"good", *DEFECTS}),
        ("ground_truth", set(DEFECTS)),
    ):
        folder = root / split
        if folder.is_dir():
            try:
                for path in folder.iterdir():
                    if path.name not in allowed:
                        issues.append(f"Unexpected entry: {split}/{path.name}")
            except OSError as error:
                issues.append(f"Cannot read {split}: {error}")

    for defect in DEFECTS:
        expected_masks = {f"{stem}_mask" for stem in images[f"test/{defect}"]}
        actual_masks = images[f"ground_truth/{defect}"]
        for stem in sorted(expected_masks - actual_masks):
            issues.append(f"Missing mask: ground_truth/{defect}/{stem}.png")
        for stem in sorted(actual_masks - expected_masks):
            issues.append(f"Orphan mask: ground_truth/{defect}/{stem}.png")
    return counts, issues


def totals_from_counts(counts):
    """Keep unavailable counts distinct from zero."""
    defects = [counts[f"test/{name}"] for name in DEFECTS]
    return {
        "normal training": counts["train/good"],
        "normal test": counts["test/good"],
        "defective test": None if None in defects else sum(defects),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/bottle"),
                        help="Folder containing train, test, ground_truth (default: data/bottle)")
    args = parser.parse_args(argv)
    counts, issues = check_dataset(args.data_root)
    print(f"Dataset: {args.data_root.resolve()}")
    print("\nFolder counts (PNG files; masks listed separately):")
    for folder, count in counts.items():
        print(f"  {folder:<35} {count if count is not None else 'MISSING'}")
    print("\nProposal comparison:")
    for label, actual in totals_from_counts(counts).items():
        expected = EXPECTED[label]
        status = "MATCH" if actual == expected else "MISMATCH"
        print(f"  {label}: actual={actual}, proposal={expected} [{status}]")
        if actual != expected:
            issues.append(f"Proposal count mismatch: {label} (actual={actual}, expected={expected})")
    print("\nScope: folders, counts, PNG signatures and mask filenames; no pixel decoding.")
    if issues:
        print("\nFAIL:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("\nPASS: all required folders, proposal counts and mask pairs match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
