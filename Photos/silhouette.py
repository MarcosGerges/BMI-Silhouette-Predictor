import math

import cv2
import mediapipe as mp
import numpy as np


def resize_image(image, max_width=256, max_height=256):
    height, width = image.shape[:2]
    if height < width:
        img = cv2.resize(image, (max_width, math.floor(height / (width / max_width))))
    else:
        img = cv2.resize(image, (math.floor(width / (height / max_height)), max_height))
    return img


def create_sillhouette(image, debug=False):
    if image is None:
        raise FileNotFoundError("Image file not found. Please check the path.")

    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_image)
        mask = results.segmentation_mask
        condition = mask > 0.3

        silhouette = np.zeros_like(image)
        silhouette[condition] = [255, 255, 255]

        cv2.imwrite("Photos/silhouette.jpg", silhouette)

        foreground_pixels = np.where(silhouette[:, :, 0] > 0)
        if len(foreground_pixels[0]) == 0:
            raise ValueError("No silhouette pixels found. The segmentation mask may be empty or too strict.")

        ys, xs = foreground_pixels
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        padding = 10
        cropped_silhouette = silhouette[y_min - padding:y_max + padding + 1,
                                       x_min - padding:x_max + padding + 1]

        target_size = max(cropped_silhouette.shape[:2])
        centered_canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - cropped_silhouette.shape[0]) // 2
        x_offset = (target_size - cropped_silhouette.shape[1]) // 2
        centered_canvas[y_offset:y_offset + cropped_silhouette.shape[0],
                        x_offset:x_offset + cropped_silhouette.shape[1]] = cropped_silhouette

        centered_and_resized = resize_image(centered_canvas)
        cv2.imwrite("Photos/centered_silhouette.jpg", centered_and_resized)

        if debug:
            cv2.imshow("Resized Silhouette", centered_and_resized)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    break
            cv2.destroyAllWindows()

        return centered_and_resized
