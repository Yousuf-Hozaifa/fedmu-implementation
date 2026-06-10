# data/iu_dataset.py
"""
Indiana University Chest X-ray Dataset loader for Fed-MU.
Replaces toy_dataset.py for Kaggle experiments.

Dataset: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
Structure:
  indiana_reports.csv     — columns: uid, findings, impression
  indiana_projections.csv — columns: uid, filename, projection
  images/images_normalized/*.png
"""

import os
import random
import pandas as pd
from PIL import Image
from pathlib import Path


def load_iu_dataset(config: dict, seed: int = 42) -> list:
    """
    Load the Indiana University Chest X-ray dataset and return a list
    of patient dicts in the same format as toy_dataset.generate_toy_dataset().

    Each patient dict:
    {
        'patient_id': str,
        'image_path': str,          # absolute path to PNG
        'report':     str,          # concatenated findings + impression
        'label':      list[int],    # 14-dim binary (placeholder for IU)
        'node_id':    int           # assigned by data_splitter later
    }
    """
    random.seed(seed)

    data_cfg = config['data']
    image_folder  = data_cfg['image_folder']   # .../images/images_normalized/
    reports_file  = data_cfg['reports_file']   # .../indiana_reports.csv
    total_patients = data_cfg['total_patients']  # 2000

    # ── Load reports ──────────────────────────────────────────────────────
    reports_df = pd.read_csv(reports_file)

    # Normalise column names (the CSV uses 'uid', 'findings', 'impression')
    reports_df.columns = [c.strip().lower() for c in reports_df.columns]

    # Some rows have NaN findings/impression — fill with empty string
    reports_df['findings']   = reports_df['findings'].fillna('')
    reports_df['impression'] = reports_df['impression'].fillna('')
    reports_df['report']     = (reports_df['findings'] + ' ' +
                                reports_df['impression']).str.strip()

    # ── Load projections (image filenames per uid) ─────────────────────────
    proj_csv = os.path.join(os.path.dirname(reports_file),
                            'indiana_projections.csv')
    proj_df  = pd.read_csv(proj_csv)
    proj_df.columns = [c.strip().lower() for c in proj_df.columns]

    # Keep only frontal projections to avoid duplicates per patient
    frontal = proj_df[proj_df['projection'].str.lower().str.contains(
        'frontal|pa|ap', na=False)]

    # Merge reports + frontal images on uid
    merged = pd.merge(reports_df, frontal[['uid', 'filename']],
                      on='uid', how='inner')

    # Drop rows where image file doesn't actually exist on disk
    def image_exists(filename):
        return os.path.isfile(os.path.join(image_folder, filename))

    merged = merged[merged['filename'].apply(image_exists)].reset_index(drop=True)

    if len(merged) == 0:
        raise FileNotFoundError(
            f"No images found in {image_folder}. "
            "Check that the dataset is mounted correctly at "
            "/kaggle/input/chest-xrays-indiana-university/")

    print(f"[IU Dataset] Found {len(merged)} valid frontal chest X-rays")

    # ── Sample up to total_patients ────────────────────────────────────────
    if len(merged) > total_patients:
        merged = merged.sample(n=total_patients, random_state=seed).reset_index(drop=True)
    else:
        print(f"[IU Dataset] WARNING: only {len(merged)} patients available, "
              f"requested {total_patients}. Using all available.")

    # ── Build patient list ─────────────────────────────────────────────────
    patients = []
    for idx, row in merged.iterrows():
        patient = {
            'patient_id': f"iu_{row['uid']:05d}" if isinstance(row['uid'], int)
                          else f"iu_{str(row['uid'])}",
            'image_path': os.path.join(image_folder, row['filename']),
            'report':     row['report'] if row['report'] else
                          "No findings reported.",
            # IU dataset has no 14-class labels — use a zero vector
            # (for training the MLP projection; labels are not used in
            #  the unlearning algorithm itself)
            'label':      [0] * 14,
            'node_id':    -1,   # assigned by data_splitter
        }
        patients.append(patient)

    print(f"[IU Dataset] Loaded {len(patients)} patients")
    return patients


def verify_dataset(config: dict) -> bool:
    """Quick sanity check — call before running the full pipeline."""
    try:
        image_folder = config['data']['image_folder']
        reports_file = config['data']['reports_file']

        assert os.path.isdir(image_folder), \
            f"Image folder not found: {image_folder}"
        assert os.path.isfile(reports_file), \
            f"Reports CSV not found: {reports_file}"

        # Count images
        images = list(Path(image_folder).glob("*.png"))
        print(f"[Verify] ✓ Image folder: {len(images)} PNG files")
        print(f"[Verify] ✓ Reports CSV:  {reports_file}")

        # Check CSV columns
        df = pd.read_csv(reports_file, nrows=5)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {'uid', 'findings', 'impression'}
        assert required.issubset(set(df.columns)), \
            f"Missing columns. Found: {list(df.columns)}"
        print(f"[Verify] ✓ CSV columns OK: {list(df.columns)}")
        return True

    except Exception as e:
        print(f"[Verify] ✗ {e}")
        return False