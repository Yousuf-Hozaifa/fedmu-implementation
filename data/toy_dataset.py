"""
data/toy_dataset.py

Generates a synthetic toy dataset that mirrors the MIMIC-CXR structure.
Each 'patient' has:
  - A synthetic chest X-ray image (random noise with pathology-like patterns)
  - A short radiology report (templated text)
  - 14 binary pathology labels
  - A unique patient ID

Why synthetic? Lets us build and validate all pipeline logic before
real data access is obtained. The structure is identical to MIMIC-CXR.
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


# The 14 MIMIC-CXR pathology labels (same as real dataset)
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# Templated radiology report fragments
FINDINGS_TEMPLATES = [
    "The lungs are clear. No focal consolidation, pleural effusion, or pneumothorax.",
    "There is a patchy opacity in the right lower lobe consistent with pneumonia.",
    "Mild cardiomegaly is present. No acute pulmonary process identified.",
    "Small bilateral pleural effusions are noted. Mild pulmonary vascular congestion.",
    "The cardiomediastinal silhouette is enlarged. Interstitial markings are increased.",
    "No acute cardiopulmonary abnormality. Heart size is normal.",
    "Left lower lobe atelectasis is present. No pneumothorax identified.",
    "Diffuse bilateral opacities consistent with pulmonary edema.",
]

IMPRESSION_TEMPLATES = [
    "No acute cardiopulmonary process.",
    "Right lower lobe pneumonia. Clinical correlation recommended.",
    "Cardiomegaly without acute pulmonary edema.",
    "Bilateral pleural effusions with pulmonary vascular congestion.",
    "Enlarged cardiac silhouette. Pulmonary vascular congestion.",
    "Normal chest radiograph.",
    "Left lower lobe atelectasis.",
    "Pulmonary edema. Recommend clinical correlation.",
]


def make_synthetic_xray(patient_id: int, has_opacity: bool = False) -> Image.Image:
    """
    Generate a synthetic grayscale chest X-ray image (224x224).

    Real X-rays are grayscale with rib structures and lung fields.
    We approximate this with structured noise: dark lung fields on the
    sides, lighter mediastinum in the centre, and an optional bright
    patch simulating an opacity/pathology.

    Returns a PIL Image in RGB mode (CLIP expects RGB).
    """
    rng = np.random.RandomState(patient_id)  # deterministic per patient

    # Base: medium-gray background (chest wall / bones)
    img = rng.randint(80, 120, (224, 224), dtype=np.uint8)

    # Lung fields: darker ovals on left and right
    for cx, cy, rx, ry in [(70, 112, 45, 70), (154, 112, 45, 70)]:
        for y in range(224):
            for x in range(224):
                if ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2) < 1:
                    img[y, x] = rng.randint(20, 50)  # dark lung field

    # Mediastinum: lighter vertical band in centre
    img[:, 90:134] = np.clip(img[:, 90:134] + rng.randint(30, 60), 0, 255)

    # Pathology: bright patch in right lower lung (if pathology present)
    if has_opacity:
        opacity_region = img[130:170, 100:150]
        img[130:170, 100:150] = np.clip(opacity_region + rng.randint(60, 90), 0, 255)

    # Add patient-specific noise (identity signal — important for E6 probe)
    # Each patient has a unique subtle noise pattern — this is what A× must forget
    identity_noise = rng.randint(-15, 15, (224, 224))
    img = np.clip(img.astype(int) + identity_noise, 0, 255).astype(np.uint8)

    # Convert grayscale to RGB (CLIP requires 3 channels)
    rgb = np.stack([img, img, img], axis=-1)
    return Image.fromarray(rgb, mode='RGB')


def make_report(patient_id: int, primary_pathology_idx: int) -> str:
    """
    Generate a templated radiology report for a patient.

    The report is deterministic given patient_id so experiments are
    reproducible across runs.
    """
    rng = random.Random(patient_id)
    findings = rng.choice(FINDINGS_TEMPLATES)
    impression = IMPRESSION_TEMPLATES[primary_pathology_idx % len(IMPRESSION_TEMPLATES)]
    pathology_name = PATHOLOGY_LABELS[primary_pathology_idx]

    return (
        f"FINDINGS: {findings} "
        f"IMPRESSION: {impression} "
        f"Primary finding: {pathology_name}."
    )


def generate_toy_dataset(
    output_dir: str = "data/toy_patients",
    n_patients: int = 50,
    seed: int = 42,
) -> dict:
    """
    Generate and save a toy dataset of n_patients synthetic patients.

    Creates:
        data/toy_patients/
            images/patient_XXXX.png
            metadata.json   ← maps patient_id → {report, labels, image_path}

    Returns the metadata dict.
    """
    random.seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    img_path = out_path / "images"
    img_path.mkdir(parents=True, exist_ok=True)

    metadata = {}

    print(f"Generating {n_patients} synthetic patients...")

    for i in range(n_patients):
        patient_id = f"patient_{i:04d}"

        # Assign a primary pathology (determines report content and image pattern)
        primary_idx = i % len(PATHOLOGY_LABELS)
        has_opacity = primary_idx in [0, 2, 7, 11]  # Atelectasis, Consolidation, Opacity, Pneumonia

        # Generate image
        image = make_synthetic_xray(patient_id=i, has_opacity=has_opacity)
        image_filename = f"{patient_id}.png"
        image.save(img_path / image_filename)

        # Generate labels (14-dim binary vector; primary label = 1)
        labels = [0] * 14
        labels[primary_idx] = 1

        # Generate report
        report = make_report(patient_id=i, primary_pathology_idx=primary_idx)

        metadata[patient_id] = {
            "patient_id": patient_id,
            "patient_idx": i,
            "image_path": f"images/{image_filename}",
            "report": report,
            "labels": labels,
            "primary_pathology": PATHOLOGY_LABELS[primary_idx],
            "has_opacity": has_opacity,
        }

    # Save metadata
    meta_path = out_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {n_patients} patients to {output_dir}/")
    print(f"  Metadata: {meta_path}")
    print(f"  Images:   {img_path}/")
    print(f"  Label distribution:")
    for label in PATHOLOGY_LABELS:
        count = sum(1 for p in metadata.values() if p["primary_pathology"] == label)
        if count > 0:
            print(f"    {label:35s}: {count}")

    return metadata

  
if __name__ == "__main__":
    meta = generate_toy_dataset(n_patients=50)
    print(f"\nSample patient:")
    sample = list(meta.values())[0]
    for k, v in sample.items():
        if k != "labels":
            print(f"  {k}: {v}")
    print(f"  labels (first 5): {sample['labels'][:5]}")


def ensure_toy_dataset(output_dir: str = "data/toy_patients",
                       n_patients: int = 50, seed: int = 42) -> None:
    """Generate the toy dataset unless metadata AND all images already exist.

    metadata.json is committed to git but images/*.png are git-ignored, so
    checking only for metadata.json made fresh clones crash on missing images.
    """
    import json
    out = Path(output_dir)
    meta = out / "metadata.json"
    if meta.exists():
        with open(meta, encoding="utf-8") as f:
            records = json.load(f)
        if all((out / r["image_path"]).exists() for r in records.values()):
            print("Toy dataset found - skipping generation.")
            return
        print("Toy metadata found but images are missing - regenerating.")
    generate_toy_dataset(output_dir, n_patients, seed)
