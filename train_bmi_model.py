import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

PROJECT_DIR = Path("/Users/admin/Desktop/Coding/Python/BMI")
DATA_DIR = PROJECT_DIR / "train"
MASK_DIR = DATA_DIR / "mask_clean"
CACHE_PATH = DATA_DIR / "cached_features_and_images.pt"
MODEL_SAVE_PATH = PROJECT_DIR / "bmi_model.pth"
CHECKPOINT_SAVE_PATH = PROJECT_DIR / "bmi_model_checkpoint.pth"

def extract_geometric_features(mask):
    """
    Extract 14 scale-invariant features:
    - normalized_ratio: area / (height^2)
    - aspect_ratio: width / height
    - density: area / (width * height)
    - 11 slice widths from 0% to 100% height, each normalized by height (search_width / height)
    Note: We DO NOT divide by max_width to preserve thickness variation across subjects!
    """
    y, x = np.where(mask > 127)
    if len(x) == 0:
        return None
    area = len(x)
    y_min, y_max = y.min(), y.max()
    height = y_max - y_min + 1
    x_min, x_max = x.min(), x.max()
    width = x_max - x_min + 1

    normalized_ratio = float(area / (height ** 2))
    aspect_ratio = float(width / height)
    density = float(area / (width * height))

    def get_largest_segment_width(row, gap_threshold=5):
        x_vals = np.where(row > 127)[0]
        if len(x_vals) == 0:
            return 0
        gaps = np.diff(x_vals)
        break_indices = np.where(gaps > gap_threshold)[0]
        segments = np.split(x_vals, break_indices + 1)
        largest = max(segments, key=len)
        return float(largest[-1] - largest[0] + 1)

    width_values = []
    for i in range(0, 101, 10):
        search_level = int(np.clip(y_min + (y_max - y_min) * (i / 100.0), 0, mask.shape[0] - 1))
        row = mask[search_level]
        w = get_largest_segment_width(row)
        width_values.append(float(w / height))

    return [normalized_ratio, aspect_ratio, density] + width_values

def load_or_create_cache(records):
    if CACHE_PATH.exists():
        print(f"Loading cached dataset from {CACHE_PATH}...")
        data = torch.load(CACHE_PATH, map_location="cpu")
        return data["images"], data["geoms"], data["bmis"], data["subjects"], data["paths"]

    print(f"Extracting features for {len(records)} images...")
    start_t = time.time()
    images = []
    geoms = []
    bmis = []
    subjects = []
    valid_paths = []

    for idx, r in enumerate(records):
        p = Path(r["path"])
        if not p.exists():
            continue
        img = Image.open(p).convert("L")
        # Resize to 128x128 for rapid, robust training
        img_128 = img.resize((128, 128), Image.Resampling.NEAREST)
        img_np = np.array(img_128, dtype=np.uint8)

        feat = extract_geometric_features(img_np)
        if feat is None:
            continue

        images.append(img_np)
        geoms.append(feat)
        bmis.append(r["bmi"])
        subjects.append(r["subject_id"])
        valid_paths.append(str(p))

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(records):
            elapsed = time.time() - start_t
            print(f"  Processed {idx + 1}/{len(records)} images ({elapsed:.1f}s)")

    images = torch.from_numpy(np.stack(images))  # [N, 128, 128] uint8
    geoms = torch.tensor(geoms, dtype=torch.float32)  # [N, 14]
    bmis = torch.tensor(bmis, dtype=torch.float32)    # [N]

    print(f"Saving cache to {CACHE_PATH} ({images.element_size() * images.nelement() / 1e6:.1f} MB)...")
    torch.save({
        "images": images,
        "geoms": geoms,
        "bmis": bmis,
        "subjects": subjects,
        "paths": valid_paths
    }, CACHE_PATH)

    return images, geoms, bmis, subjects, valid_paths

class MemoryBMIDataset(Dataset):
    def __init__(self, images, geoms, bmis_norm):
        self.images = images.float().unsqueeze(1) / 255.0  # [N, 1, 128, 128]
        self.geoms = torch.as_tensor(geoms, dtype=torch.float32)
        self.bmis_norm = torch.as_tensor(bmis_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.bmis_norm)

    def __getitem__(self, idx):
        return self.images[idx], self.geoms[idx], self.bmis_norm[idx]

class BMIModel(nn.Module):
    def __init__(self, num_geometry_features=14):
        super().__init__()
        # CNN Branch
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> 1x1 Global Average Pool
        )

        self.cnn_fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Geometry Branch
        self.geometry_fc = nn.Sequential(
            nn.Linear(num_geometry_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Combined Regressor Head
        self.final_fc = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, image, geometry):
        x = self.cnn(image)
        x = x.flatten(1)
        x = self.cnn_fc(x)

        g = self.geometry_fc(geometry)

        combined = torch.cat((x, g), dim=1)
        output = self.final_fc(combined)
        return output

def main():
    print("=== Starting BMI Model Training Pipeline ===")
    
    # 1. Load metadata
    hwg_df = pd.read_csv(DATA_DIR / "hwg_metadata.csv")
    map_df = pd.read_csv(DATA_DIR / "subject_to_photo_map.csv")
    bmi_lookup = dict(zip(hwg_df.iloc[:, 0], hwg_df.iloc[:, -1]))

    records = []
    for _, row in map_df.iterrows():
        subj, photo = row.iloc[0], row.iloc[-1]
        if subj in bmi_lookup:
            p = MASK_DIR / f"{photo}.png"
            if p.exists():
                records.append({
                    "subject_id": subj,
                    "path": str(p),
                    "bmi": float(bmi_lookup[subj])
                })

    print(f"Found {len(records)} valid mapped photos across {len(set(r['subject_id'] for r in records))} subjects.")

    # 2. Cache / Load features and images
    images, geoms, bmis, subjects, paths = load_or_create_cache(records)
    print(f"Dataset size: {len(bmis)} samples.")

    # 3. Subject-level Group Split (80% train, 20% val)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(paths, groups=subjects))

    print(f"Train samples: {len(train_idx)} ({len(set(subjects[i] for i in train_idx))} subjects)")
    print(f"Val samples:   {len(val_idx)} ({len(set(subjects[i] for i in val_idx))} subjects)")

    train_imgs, val_imgs = images[train_idx], images[val_idx]
    train_geoms_raw, val_geoms_raw = geoms[train_idx].numpy(), geoms[val_idx].numpy()
    train_bmis_raw, val_bmis_raw = bmis[train_idx].numpy(), bmis[val_idx].numpy()

    # 4. Fit StandardScaler on Train geometry only
    scaler = StandardScaler().fit(train_geoms_raw)
    train_geoms_norm = scaler.transform(train_geoms_raw)
    val_geoms_norm = scaler.transform(val_geoms_raw)

    # 5. Target Normalization on Train BMI only
    bmi_mean = float(train_bmis_raw.mean())
    bmi_std = float(train_bmis_raw.std())
    print(f"Train BMI: mean={bmi_mean:.2f}, std={bmi_std:.2f}, min={train_bmis_raw.min():.2f}, max={train_bmis_raw.max():.2f}")
    print(f"Val BMI:   mean={val_bmis_raw.mean():.2f}, std={val_bmis_raw.std():.2f}, min={val_bmis_raw.min():.2f}, max={val_bmis_raw.max():.2f}")

    train_bmis_norm = (train_bmis_raw - bmi_mean) / bmi_std
    val_bmis_norm = (val_bmis_raw - bmi_mean) / bmi_std

    train_dataset = MemoryBMIDataset(train_imgs, train_geoms_norm, train_bmis_norm)
    val_dataset = MemoryBMIDataset(val_imgs, val_geoms_norm, val_bmis_norm)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 6. Initialize Model, Loss, Optimizer
    device = torch.device("cpu")
    model = BMIModel(num_geometry_features=14).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    epochs = 20
    best_val_mae = float("inf")
    best_state_dict = None

    print("\n--- Training Loop ---")
    start_train_t = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for b_imgs, b_geoms, b_bmis in train_loader:
            b_imgs = b_imgs.to(device)
            b_geoms = b_geoms.to(device)
            b_bmis = b_bmis.to(device)

            optimizer.zero_grad()
            preds = model(b_imgs, b_geoms).squeeze(1)
            loss = criterion(preds, b_bmis)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_list = []

        with torch.no_grad():
            for b_imgs, b_geoms, b_bmis in val_loader:
                b_imgs = b_imgs.to(device)
                b_geoms = b_geoms.to(device)
                b_bmis = b_bmis.to(device)

                preds = model(b_imgs, b_geoms).squeeze(1)
                loss = criterion(preds, b_bmis)
                val_loss += loss.item()

                val_preds_list.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Calculate actual BMI errors
        val_preds_actual = np.array(val_preds_list) * bmi_std + bmi_mean
        errors = val_preds_actual - val_bmis_raw
        mae = float(np.mean(np.abs(errors)))
        mean_err = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        scheduler.step(mae)
        epoch_dur = time.time() - epoch_start

        if mae < best_val_mae:
            best_val_mae = mae
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " (*BEST*)"
        else:
            improved = ""

        print(f"Epoch {epoch+1:02d}/{epochs} [{epoch_dur:.1f}s] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {mae:.2f} | Bias: {mean_err:+.2f}{improved}")

    total_train_dur = time.time() - start_train_t
    print(f"\nTraining completed in {total_train_dur:.1f} seconds.")

    # 7. Evaluate Best Model
    model.load_state_dict(best_state_dict)
    model.eval()

    val_preds_list = []
    with torch.no_grad():
        for b_imgs, b_geoms, _ in val_loader:
            preds = model(b_imgs.to(device), b_geoms.to(device)).squeeze(1)
            val_preds_list.extend(preds.cpu().numpy())

    val_preds_actual = np.array(val_preds_list) * bmi_std + bmi_mean
    errors = val_preds_actual - val_bmis_raw
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_bias = float(np.mean(errors))
    std_dev = float(np.std(errors))
    median_err = float(np.median(abs_errors))
    p90_err = float(np.percentile(abs_errors, 90))

    # Pearson correlation and R^2
    corr = float(np.corrcoef(val_preds_actual, val_bmis_raw)[0, 1])
    ss_res = np.sum((val_bmis_raw - val_preds_actual) ** 2)
    ss_tot = np.sum((val_bmis_raw - np.mean(val_bmis_raw)) ** 2)
    r2 = float(1 - (ss_res / ss_tot))

    print("\n==========================================")
    print("       FINAL MODEL EVALUATION RESULTS     ")
    print("==========================================")
    print(f"Validation Samples:       {len(val_bmis_raw)}")
    print(f"Mean Absolute Error (MAE): {mae:.3f} BMI points")
    print(f"Root Mean Sq Error (RMSE): {rmse:.3f} BMI points")
    print(f"Mean Error (Bias):        {mean_bias:+.3f} BMI points")
    print(f"Standard Deviation of Err:{std_dev:.3f} BMI points")
    print(f"Median Absolute Error:    {median_err:.3f} BMI points")
    print(f"90th Percentile Error:    {p90_err:.3f} BMI points")
    print(f"Pearson Correlation (r):  {corr:.3f}")
    print(f"R-squared (R^2):          {r2:.3f}")
    print("------------------------------------------")
    print("Error by BMI Category:")
    categories = [
        ("Underweight (<18.5)", val_bmis_raw < 18.5),
        ("Normal (18.5 - 25.0)", (val_bmis_raw >= 18.5) & (val_bmis_raw < 25.0)),
        ("Overweight (25.0 - 30.0)", (val_bmis_raw >= 25.0) & (val_bmis_raw < 30.0)),
        ("Obese (>=30.0)", val_bmis_raw >= 30.0),
    ]
    for label, mask in categories:
        count = int(np.sum(mask))
        if count > 0:
            cat_mae = float(np.mean(abs_errors[mask]))
            cat_bias = float(np.mean(errors[mask]))
            print(f"  {label:<24}: N={count:>4} | MAE = {cat_mae:5.2f} | Bias = {cat_bias:+5.2f}")
    print("==========================================\n")

    # 8. Save Weights and Production Checkpoint
    torch.save(best_state_dict, MODEL_SAVE_PATH)
    print(f"Saved state dict to: {MODEL_SAVE_PATH}")

    checkpoint = {
        "model_state_dict": best_state_dict,
        "num_geometry_features": 14,
        "image_size": 128,
        "bmi_mean": bmi_mean,
        "bmi_std": bmi_std,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "mean_bias": mean_bias,
            "std_dev": std_dev,
            "r2": r2,
            "corr": corr
        }
    }
    torch.save(checkpoint, CHECKPOINT_SAVE_PATH)
    print(f"Saved full production checkpoint to: {CHECKPOINT_SAVE_PATH}")

if __name__ == "__main__":
    main()
