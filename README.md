# BMI Prediction from Body Silhouette

An end-to-end computer vision and deep learning system that estimates a person's Body Mass Index (BMI) directly from a 2D silhouette captured via a standard webcam.

---

## Overview

Traditional anthropometric measurement typically requires physical scales and measuring tape. This project demonstrates how 2D body contours and dimensionless shape geometry can accurately estimate BMI using a hybrid PyTorch neural network.

The system consists of:
1. **Webcam Capture & Pose Detection**: Real-time camera feed using MediaPipe Pose for full-body tracking.
2. **Silhouette Extraction**: High-fidelity background removal with morphological filtering and proportional 10% padding.
3. **Geometric Feature Engineering**: Extracts 14 scale-invariant body thickness features (aspect ratio, normalized area ratio, density, and 11 horizontal slice widths from head to feet).
4. **Hybrid Deep Learning Model**: A fused architecture combining a 4-layer CNN (with `BatchNorm2d`, Global Average Pooling, and `Dropout`) and an MLP geometry network trained with `SmoothL1Loss`.
5. **Real-time Live Predictor**: Direct end-to-end inference from live webcam capture to console BMI reporting.

---

## Model Performance

Trained on over 6,100 silhouette masks from the BodyM dataset and validated on a held-out test split of 1,167 samples across 400 unique subjects (split strictly by `subject_id` to prevent data leakage):

| Metric | Score |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | **1.489 BMI points** |
| **Root Mean Squared Error (RMSE)** | **2.006 BMI points** |
| **Mean Error (Bias)** | **+0.060 BMI points** |
| **Median Absolute Error** | **1.191 BMI points** |
| **90th Percentile Error** | **3.037 BMI points** |
| **Pearson Correlation ($r$)** | **0.951** |
| **Coefficient of Determination ($R^2$)** | **0.892** |

---

## Project Structure

```
├── main.py                     # Entry point: runs camera capture and prints predicted BMI
├── Camera.py                   # OpenCV webcam interface with MediaPipe pose tracking
├── predict_bmi.py              # Production inference class for single images or live arrays
├── train_bmi_model.py          # Complete PyTorch training pipeline with caching & evaluation
├── Database_Photo_cleanup.py   # Dataset preprocessor: mask cropping, padding, and resizing
├── Train_data_setup.py         # Metadata alignment and mapping scripts
├── Model_Training.ipynb        # Interactive Jupyter notebook for training and analysis
├── Photos/                     # Silhouette and webcam capture directory
├── train/                      # Dataset metadata CSV files
├── bmi_model.pth               # Trained PyTorch model weights
├── bmi_model_checkpoint.pth    # Complete checkpoint (weights + normalization parameters)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MarcosGerges/BMI-Silhouette-Predictor.git
   cd BMI-Silhouette-Predictor
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Live Webcam Prediction

Run the application:
```bash
python main.py
```
- Stand 2–3 meters back from the webcam so your full body (head to feet) is in view.
- Stand in a natural upright posture with arms relaxed at your sides.
- Hold still for the 3-second countdown.
- The silhouette is automatically generated and evaluated by the neural network:

```text
==============================================
            BMI PREDICTION RESULT             
==============================================
  Photo Timestamp  : 20261027_193000
  Raw Model Output : 27.20
  Applied Offset   : -5.00
  Calibrated BMI   : 22.20
  Classification   : Normal weight
==============================================
```

### 2. Optional Calibration Offset

Wide-angle webcams can introduce perspective dilation when standing close. You can customize the calibration offset on the fly:
```bash
python main.py --offset -4.5
```
Or view the raw, unadjusted model output:
```bash
python main.py --raw
```

### 3. Python API

You can also use the predictor directly in Python:
```python
from predict_bmi import BMIPredictor

predictor = BMIPredictor("bmi_model_checkpoint.pth")
bmi = predictor.predict("path/to/silhouette.png")
print(f"Predicted BMI: {bmi:.2f}")
```

---

## Retraining the Model

To retrain the model from scratch on the BodyM dataset:
```bash
python train_bmi_model.py
```
Features and downscaled masks are cached into `train/cached_features_and_images.pt` on the first run for fast multi-core training.

---

## License

MIT License.
