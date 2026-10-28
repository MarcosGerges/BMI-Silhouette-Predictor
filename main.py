import argparse
import sys
from pathlib import Path
from Camera import take_photo
from Photos.sillhouette_creation import create_sillhouette
from predict_bmi import BMIPredictor

# ---------------------------------------------------------------------------
# CALIBRATION CONFIGURATION
# ---------------------------------------------------------------------------
# Standard offset to calibrate for webcam lens perspective and room distance.
# Default is set to -5.0 as requested. Set to 0.0 to disable offset.
DEFAULT_CALIBRATION_OFFSET = -5.0
# ---------------------------------------------------------------------------


def get_bmi_category(bmi: float) -> str:
    """Return standard WHO BMI category."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal weight"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


def main():
    parser = argparse.ArgumentParser(description="BMI Prediction from Silhouette")
    parser.add_argument(
        "--offset",
        type=float,
        default=DEFAULT_CALIBRATION_OFFSET,
        help="Calibration offset to apply to predicted BMI (default: -5.0)"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Display raw model prediction without offset"
    )
    args = parser.parse_args()

    offset = 0.0 if args.raw else args.offset

    checkpoint_path = Path(__file__).resolve().parent / "bmi_model_checkpoint.pth"
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print("Loading BMI prediction model...")
    predictor = BMIPredictor(checkpoint_path=checkpoint_path)
    print("Model ready. Launching camera...\n")

    capture_result = take_photo()
    if capture_result is None:
        print("\nNo photo was captured. Exiting.")
        return

    image, photo_timestamp = capture_result

    print(f"\nPhoto captured ({photo_timestamp}). Creating silhouette...")
    try:
        silhouette = create_sillhouette(image, photo_timestamp)
    except Exception as e:
        print(f"Error creating silhouette: {e}")
        return

    if silhouette is None:
        print("Failed to generate silhouette from photo.")
        return

    print("Silhouette created successfully. Feeding to model...")
    try:
        raw_bmi = predictor.predict(silhouette)
        calibrated_bmi = raw_bmi + offset
        category = get_bmi_category(calibrated_bmi)

        print("\n" + "=" * 46)
        print("            BMI PREDICTION RESULT             ")
        print("=" * 46)
        print(f"  Photo Timestamp  : {photo_timestamp}")
        print(f"  Raw Model Output : {raw_bmi:.2f}")
        if offset != 0.0:
            print(f"  Applied Offset   : {offset:+.2f}")
            print(f"  Calibrated BMI   : {calibrated_bmi:.2f}")
        else:
            print(f"  Predicted BMI    : {raw_bmi:.2f}")
        print(f"  Classification   : {category}")
        print("=" * 46 + "\n")

    except Exception as e:
        print(f"Error during BMI prediction: {e}")


if __name__ == "__main__":
    main()
