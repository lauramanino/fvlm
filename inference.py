#!/usr/bin/env python3
"""
FVLM Inference for CT scans (DICOM zips → CSV + stdout table).

Usage:
    python inference.py scan1.zip scan2.zip
    python inference.py scan*.zip          # all scans
    python inference.py scan1.zip --cpu    # force CPU
    python inference.py scan1.zip --no-mask  # skip TotalSegmentator
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import pydicom
from tqdm import tqdm

from monai import transforms

# Suppress TotalSegmentator warnings during batch runs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from lavis.common.config import Config
from lavis.common.registry import registry

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ORGANS = ["lung", "heart", "esophagus", "aorta"]

# Broad label mapping: TotalSegmentator uses labels in the 1–120+ range.
# We map them to 4 organ IDs: 1=lung, 2=heart, 3=esophagus, 4=aorta.
# Exact label numbers vary across TS versions; ranges catch the dominant ones.

# Build ORGAN_LABEL_MAP from actual TotalSegmentator label values.
# We scan the mask data and assign organ IDs based on label ranges,
# prioritizing aorta (13-14) before heart absorbs them.
def _build_label_map(mask_data):
    unique = set(int(x) for x in np.unique(mask_data) if x > 0)
    label_map = {}
    # Aorta first (labels 13-14) so they don't get absorbed by heart
    for l in unique:
        if l in (13, 14):
            label_map[l] = 4  # aorta
    # Then heart (labels 5-6, 10-12)
    for l in unique:
        if l in (5, 6) or (10 <= l <= 12):
            label_map[l] = 2  # heart
    # Lung (labels 1-4)
    for l in unique:
        if l in range(1, 5):
            label_map[l] = 1  # lung
    # Esophagus (labels 15-25)
    for l in unique:
        if 15 <= l <= 25:
            label_map[l] = 3  # esophagus
    # Fallback: any remaining label gets organ 2 (heart) if significant, else 1 (lung)
    for l in unique:
        if l not in label_map:
            label_map[l] = 1  # default to lung
    return label_map

ORGAN_LABEL_MAP = {}  # populated dynamically in generate_organ_mask

# FVLM test items: (organ, disease, neg_text, pos_text)
TEST_ITEMS = [
    ("lung", "Emphysema", "Not Emphysema.", "Emphysema."),
    ("lung", "Atelectasis", "Not Atelectatic.", "Atelectatic."),
    ("lung", "Lung nodule", "Not Nodule.", "Nodule."),
    ("lung", "Lung opacity", "Not Opacity.", "Opacity."),
    ("lung", "Pulmonary fibrotic sequela", "Not Pulmonary fibrotic.", "Pulmonary fibrotic."),
    ("lung", "Pleural effusion", "Not Pleural effusion.", "Pleural effusion."),
    ("lung", "Mosaic attenuation pattern", "Not Mosaic attenuation pattern.", "Mosaic attenuation pattern."),
    ("lung", "Peribronchial thickening", "Not Peribronchial thickening.", "Peribronchial thickening."),
    ("lung", "Consolidation", "Not Consolidation.", "Consolidation."),
    ("lung", "Bronchiectasis", "Not Bronchiectasis.", "Bronchiectasis."),
    ("lung", "Interlobular septal thickening", "Not Interlobular septal thickening.", "Interlobular septal thickening."),
    ("heart", "Cardiomegaly", "Not Cardiomegaly.", "Cardiomegaly."),
    ("heart", "Pericardial effusion", "Not Pericardial effusion.", "Pericardial effusion."),
    ("heart", "Coronary artery wall calcification", "Not Coronary artery wall calcification.", "Coronary artery wall calcification."),
    ("esophagus", "Hiatal hernia", "Not Hiatal hernia.", "Hiatal hernia."),
    ("aorta", "Arterial wall calcification", "Not Arterial wall calcification.", "Arterial wall calcification."),
]

CFG_PATH = "lavis/projects/blip/train/pretrain_ct.yaml"
CKPT_PATH = "model.pth"


# ──────────────────────────────────────────────────────────────────────────────
# DICOM Reading
# ──────────────────────────────────────────────────────────────────────────────

def find_dicom_files(exam_dir: str):
    """
    Find all DICOM files in the extracted zip's exam directory.
    Returns a dict of {series_uid: [(filepath, pydicom_dataset), ...]}.
    Skips JPEG subdirectories and non-CT files.
    """
    series_map = defaultdict(list)
    skip_dirs = {"jpeg", "ddv"}

    for root, dirs, files in os.walk(exam_dir):
        # Skip known non-DICOM subdirectories
        if any(s in root.lower() for s in skip_dirs):
            continue

        for fname in files:
            fpath = os.path.join(root, fname)
            # Skip metadata files
            if fname in ("DICOMDIR", "VERSION", "LOCKFILE", "mv.ini"):
                continue
            # Try to read as DICOM
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=False)
                if ds.Modality == "CT":
                    series_map[ds.SeriesInstanceUID].append((fpath, ds))
            except Exception:
                continue

    return series_map


def subsample_volume(volume: np.ndarray, max_slices: int = 100) -> np.ndarray:
    """Subsample volume slices to reduce memory for ViT processing."""
    if volume.shape[0] <= max_slices:
        return volume
    stride = max(1, volume.shape[0] // max_slices)
    return volume[::stride, :, :]
def build_volume_from_dicom(series_map: dict, device: torch.device):
    """
    From a series map, pick the largest CT series, sort by instance number,
    and build a 3D volume tensor [1, C, D, H, W].
    """
    # Pick largest series
    largest_uid = max(series_map, key=lambda uid: len(series_map[uid]))
    series_files = series_map[largest_uid]

    # Sort by instance number
    def get_instance_num(item):
        ds = item[1]
        if hasattr(ds, "InstanceNumber") and ds.InstanceNumber:
            return int(ds.InstanceNumber)
        return 0

    series_files.sort(key=get_instance_num)
    print(f"  Series UID: {largest_uid[:60]}... ({len(series_files)} slices)")

    # Build volume
    ds0 = series_files[0][1]
    rows, cols = ds0.Rows, ds0.Columns
    volume = np.zeros((len(series_files), rows, cols), dtype=np.float32)

    for i, (fpath, ds) in enumerate(series_files):
        arr = ds.pixel_array
        if arr.ndim == 3:  # handle multi-frame
            arr = arr[0]
        # Convert to HU if rescale parameters present
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        volume[i] = arr.astype(np.float32) * slope + intercept

    print(f"  Volume shape: {volume.shape} (slices × H × W)")
    print(f"  Pixel range: {volume.min():.1f} to {volume.max():.1f} HU")

    # Create NIfTI image
    affine = np.eye(4)
    if hasattr(ds0, "PixelSpacing") and ds0.PixelSpacing:
        ps = ds0.PixelSpacing
        affine[0, 0] = ps[0] if isinstance(ps[0], (int, float)) else float(ps[0])
        affine[1, 1] = ps[1] if isinstance(ps[1], (int, float)) else float(ps[1])
    # Subsample volume for memory efficiency during ViT processing
    volume = subsample_volume(volume, max_slices=100)
    
    nifti_img = nib.Nifti1Image(volume, affine)

    # Save to temp NIfTI for TotalSegmentator
    nifti_path = os.path.join(tempfile.gettempdir(), "fvlm_temp_volume.nii.gz")
    nib.save(nifti_img, nifti_path)
    print(f"  NIfTI saved: {nifti_path}")

    return nifti_path


# ──────────────────────────────────────────────────────────────────────────────
# Mask Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_organ_mask(nifti_path: str, device: torch.device) -> torch.Tensor:
    """
    Use TotalSegmentator to generate a 4-organ segmentation mask.
    Returns a [1, D, H, W] tensor with values 1-4.
    """
    from totalsegmentator.python_api import totalsegmentator

    print("  Running TotalSegmentator for organ mask generation…")
    input_img = nib.load(nifti_path)
    mask_img = totalsegmentator(input_img, quiet=True, output_type="nifti", device="cpu", fast=True)
    mask_data = mask_img.get_fdata()
    torch.cuda.empty_cache()  # Free any cached GPU memory (no-op on CPU)
    mask_np = mask_data.astype(np.float32)

    # Create simplified 4-class mask
    organ_mask = np.zeros_like(mask_np, dtype=np.float32)

    # Map TotalSegmentator labels to our 4 organs
    _label_map = _build_label_map(mask_data)
    ORGAN_LABEL_MAP.update(_label_map)
    for ts_label, organ_val in _label_map.items():
        organ_mask[mask_data == ts_label] = organ_val

    # Fallback: if no organs detected, check what labels exist
    unique_labels = np.unique(mask_data)
    if organ_mask.max() == 0:
        print(f"  ⚠️  Standard label mapping returned zeros. Unique labels: {unique_labels[:20]}")
        # Try a broader mapping
        for label in unique_labels:
            if label > 0:
                organ_mask[mask_data == label] = 1  # default to lung
                break
    # Use uint8 for mask to reduce memory; move to GPU
    mask_tensor = torch.from_numpy(organ_mask.astype(np.uint8)).unsqueeze(0).to(device)
    unique_vals = torch.unique(mask_tensor[0])
    print(f"  Generated mask. Unique values: {unique_vals.tolist()}")
    return mask_tensor


def fallback_mask(volume_shape, device: torch.device) -> torch.Tensor:
    """Create a simple all-lung mask if TotalSegmentator fails."""
    mask = torch.ones((1, *volume_shape[2:]), device=device, dtype=torch.float32)
    print("  Using fallback all-lung mask.")
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading & Inference
# ──────────────────────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    """Load the FVLM model from config and checkpoint."""
    cfg = Config(type("Args", (), {"cfg_path": CFG_PATH, "options": None})())
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    print(f"Loading weights from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    device: torch.device,
    skip_organ: int = 0,
):
    """
    Run FVLM inference on the given image and mask.
    Returns a dict of (organ, disease) → probability.
    """
    active_items = [item for item in TEST_ITEMS if item[0] in ORGANS]

    # Compute organ sizes from mask
    whole_organ_sizes = {
        org: torch.eq(mask_tensor, ORGANS.index(org) + 1).sum().item()
        for org in ORGANS
    }
    active_organs = [org for org in ORGANS if whole_organ_sizes[org] > 0]
    active_items = [item for item in active_items if item[0] in active_organs]

    if not active_items:
        print("  No active organs found in mask.")
        return {}

    # Text feature extraction
    print("  Extracting text features…")
    text_feat_dict = model.prepare_text_feat(active_items)
    organ_feat_dict = {}
    organ_logits = {item: [] for item in active_items}

    # Run inference
    print("  Running model inference…")
    with torch.no_grad():
        organ_logits = model.forward_test_win(
            image_tensor,
            mask_tensor,
            organ_logits,
            active_organs,
            text_feat_dict,
            organ_feat_dict,
            whole_organ_sizes,
            skip_organ=skip_organ,
        )

    # Aggregate results
    results = {}
    for item, probs in organ_logits.items():
        if len(probs) > 0:
            prob_positive = np.concatenate(probs).mean(0)[1]
            results[(item[0], item[1])] = prob_positive

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_table(all_results: dict):
    """Format results as a pretty stdout table."""
    header = f"{'Scan':<12}"
    for organ, disease, *_ in TEST_ITEMS:
        header += f" | {organ:8} {disease:32}"
    border = "=" * len(header)

    print("\n" + border)
    print(header)
    print(border)

    for scan_name, scan_results in all_results.items():
        row = f"{scan_name:<12}"
        for organ, disease, *_ in TEST_ITEMS:
            key = (organ, disease)
            if key in scan_results:
                prob = scan_results[key]
                if prob >= 0.7:
                    marker = "✓"
                elif prob >= 0.3:
                    marker = "≈"
                else:
                    marker = "✗"
                row += f" | {prob:6.3f}{marker}"
            else:
                row += f" | {'—':>8}"
        print(row)

    print(border + "\n")


def write_csv(all_results: dict, output_path: str = "results.csv"):
    """Write results to a CSV file."""
    disease_cols = [f"{organ}_{disease.replace(' ', '_')}" for organ, disease, *_ in TEST_ITEMS]
    header = ["scan"] + disease_cols

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for scan_name, scan_results in all_results.items():
            row = [scan_name]
            for organ, disease, *_ in TEST_ITEMS:
                key = (organ, disease)
                row.append(f"{scan_results.get(key, ''):.4f}" if key in scan_results else "")
            writer.writerow(row)

    print(f"Results saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Processing
# ──────────────────────────────────────────────────────────────────────────────

def extract_zip(zip_path: str, label: str = "") -> str:
    """Extract a .zip file to a temporary directory."""
    tmpdir = tempfile.mkdtemp(prefix=f"fvlm_{label}_")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    return tmpdir


def process_scan(
    scan_path: str,
    device: torch.device,
    use_mask: bool = True,
) -> dict:
    """Process a single scan (.zip or directory) and return results."""
    scan_name = Path(scan_path).stem
    print(f"\n{'─' * 60}")
    print(f"Processing: {scan_name}")
    print(f"{'─' * 60}")

    tmpdir = None
    exam_dir = None

    try:
        if str(scan_path).endswith(".zip"):
            tmpdir = extract_zip(scan_path, label=scan_name)
            exam_dir = os.path.join(tmpdir, "exam")
        else:
            exam_dir = str(scan_path)

        if not os.path.isdir(exam_dir):
            print(f"  ⚠️  exam directory not found: {exam_dir}")
            return {}

        # Step 1: Read DICOM files
        print(f"  Reading DICOM files from: {exam_dir}")
        series_map = find_dicom_files(exam_dir)
        if not series_map:
            print("  No DICOM files found!")
            return {}

        # Step 2: Build volume → NIfTI
        nifti_path = build_volume_from_dicom(series_map, device)

        # Step 3: Load model
        model = load_model(device)

        # Step 4: Generate mask
        if use_mask:
            mask_tensor = generate_organ_mask(nifti_path, device)
        else:
            # Load volume shape for fallback
            nifti_img = nib.load(nifti_path)
            mask_tensor = torch.ones(
                (1, *nifti_img.shape), device=device, dtype=torch.float32
            )

        # Load volume tensor for inference
        vol_img = nib.load(nifti_path)
        image_tensor = torch.from_numpy(vol_img.get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        # Step 5: Run inference
        results = run_inference(model, image_tensor, mask_tensor, device)

        # Step 6: Display results
        print(f"\n  Active organs in {scan_name}:")
        for org in ORGANS:
            count = torch.eq(mask_tensor, ORGANS.index(org) + 1).sum().item()
            print(f"    {org:12} → {count:>8} voxels")

        print(f"\n  Results for {scan_name}:")
        for (organ, disease), prob in results.items():
            marker = "✓" if prob >= 0.7 else ("≈" if prob >= 0.3 else "✗")
            print(f"    {organ:12} | {disease:32} → {prob:.4f} {marker}")

        return results

    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        # Cleanup temp NIfTI
        nifti_path = os.path.join(tempfile.gettempdir(), "fvlm_temp_volume.nii.gz")
        if os.path.exists(nifti_path):
            os.remove(nifti_path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FVLM Inference for CT DICOM scans (zips → CSV + table)",
    )
    parser.add_argument(
        "scans",
        nargs="+",
        help="Scan files (.zip) or directories. Example: scan1.zip scan2.zip",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results.csv",
        help="Output CSV file path (default: results.csv)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Skip TotalSegmentator mask generation (use all-lung mask).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (no CUDA).",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device}")

    all_results = {}
    for scan_path in args.scans:
        if not os.path.exists(scan_path):
            print(f"⚠️  Skipping {scan_path} (not found)")
            continue
        results = process_scan(scan_path, device, use_mask=not args.no_mask)
        all_results[Path(scan_path).stem] = results

    if all_results:
        format_table(all_results)
        write_csv(all_results, output_path=args.output)
    else:
        print("No scans processed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
