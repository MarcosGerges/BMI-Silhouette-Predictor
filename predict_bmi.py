import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_DIR = Path("/Users/admin/Desktop/Coding/Python/BMI")
CHECKPOINT_PATH = PROJECT_DIR / "bmi_model_checkpoint.pth"

class BMIModel(nn.Module):
    def __init__(self, num_geometry_features=14):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.cnn_fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.geometry_fc = nn.Sequential(
            nn.Linear(num_geometry_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

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
        return self.final_fc(combined)

def extract_geometric_features(mask):
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

class BMIPredictor:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.device = torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.bmi_mean = ckpt["bmi_mean"]
        self.bmi_std = ckpt["bmi_std"]
        self.scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
        self.image_size = ckpt.get("image_size", 128)

        self.model = BMIModel(num_geometry_features=ckpt["num_geometry_features"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, image_or_path):
        """
        Accepts a file path (str or Path) or a numpy array (grayscale silhouette).
        Returns predicted BMI float.
        """
        if isinstance(image_or_path, (str, Path)):
            pil_img = Image.open(image_or_path).convert("L")
        elif isinstance(image_or_path, np.ndarray):
            if len(image_or_path.shape) == 3:
                import cv2
                gray = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_or_path
            pil_img = Image.fromarray(gray)
        else:
            raise TypeError("Expected file path or numpy array")

        # Resize for inference
        img_resized = pil_img.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        img_np = np.array(img_resized, dtype=np.float32)

        raw_geom = extract_geometric_features(img_np)
        if raw_geom is None:
            raise ValueError("No silhouette foreground detected in image.")

        # Standardize geometry features
        geom_norm = (np.array(raw_geom, dtype=np.float32) - self.scaler_mean) / self.scaler_scale

        img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        geom_tensor = torch.tensor(geom_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_norm = self.model(img_tensor, geom_tensor).item()

        # Denormalize to actual BMI
        bmi_pred = pred_norm * self.bmi_std + self.bmi_mean
        return float(bmi_pred)

if __name__ == "__main__":
    predictor = BMIPredictor()
    test_img = PROJECT_DIR / "train/mask_clean/0b8fa4d88d8a238a483d382b8f787372.png"
    if test_img.exists():
        predicted = predictor.predict(test_img)
        print(f"Sample: {test_img.name}")
        print(f"Predicted BMI: {predicted:.2f}")
