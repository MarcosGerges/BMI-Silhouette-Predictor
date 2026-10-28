import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


def create_sillhouette(image, timestamp, debug=False):
    """
    Generate a clean binary silhouette matching training data specifications:
    1. Uses MediaPipe Pose segmentation.
    2. Morphological filtering to eliminate noise.
    3. Largest connected component selection (dropping background artifacts).
    4. Exact 10% proportional padding and centering to 256x256.
    """
    if image is None:
        raise FileNotFoundError("Image file not found. Please check the path.")

    h, w = image.shape[:2]
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_image)

        if not pose_results.pose_landmarks or pose_results.segmentation_mask is None:
            raise ValueError("No full body detected in frame.")

        seg_mask = pose_results.segmentation_mask

    # 1. Clean thresholding (0.60 threshold gives crisp body boundaries)
    raw_binary = np.uint8((seg_mask > 0.60) * 255)

    # 2. Morphological cleanup: remove small floating noise specs and close internal gaps
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(raw_binary, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    # 3. Connected components: keep strictly the largest foreground object (the person)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        person_mask = np.uint8((labels == largest_label) * 255)
    else:
        person_mask = cleaned

    # 4. Proportional cropping and centering (exact Database_Photo_cleanup.py logic)
    foreground_y, foreground_x = np.where(person_mask > 127)
    if len(foreground_x) == 0:
        raise ValueError("No detectable silhouette foreground found after cleaning.")

    x_min, x_max = foreground_x.min(), foreground_x.max()
    y_min, y_max = foreground_y.min(), foreground_y.max()

    person_w = x_max - x_min + 1
    person_h = y_max - y_min + 1

    padding_x = int(person_w * 0.10)
    padding_y = int(person_h * 0.10)

    x_start = max(0, x_min - padding_x)
    x_end = min(w, x_max + padding_x + 1)
    y_start = max(0, y_min - padding_y)
    y_end = min(h, y_max + padding_y + 1)

    cropped = person_mask[y_start:y_end, x_start:x_end]

    crop_h, crop_w = cropped.shape[:2]
    canvas_size = max(crop_h, crop_w)

    top = (canvas_size - crop_h) // 2
    bottom = canvas_size - crop_h - top
    left = (canvas_size - crop_w) // 2
    right = canvas_size - crop_w - left

    centered_canvas = cv2.copyMakeBorder(
        cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    centered_and_resized = cv2.resize(
        centered_canvas, (256, 256), interpolation=cv2.INTER_NEAREST
    )

    photos_dir = Path("Photos")
    photos_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(photos_dir / f"centered_silhouette_{timestamp}.jpg"), centered_and_resized)
    cv2.imwrite(str(photos_dir / "silhouette.jpg"), centered_and_resized)

    if debug:
        cv2.imshow("Silhouette", centered_and_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return centered_and_resized
