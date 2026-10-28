from pathlib import Path

import cv2
import numpy as np


INPUT_DIR = Path('/Users/admin/BodyM/train/mask')
OUTPUT_DIR = Path('/Users/admin/BodyM/train/mask_clean')
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
OUTPUT_SIZE = 256
PADDING_RATIO = 0.1


def crop_and_resize_person(image):
    """Crop a binary person mask, preserve proportions, and resize to 256x256."""

    image_height, image_width = image.shape[:2]

    # Convert to grayscale
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image

    # Find the white/foreground pixels
    foreground_y, foreground_x = np.where(grayscale > 127)

    # No person detected
    if len(foreground_x) == 0:
        return None

    # Bounding box around the person
    x_min = foreground_x.min()
    x_max = foreground_x.max()
    y_min = foreground_y.min()
    y_max = foreground_y.max()

    # Bounding box dimensions
    person_width = x_max - x_min + 1
    person_height = y_max - y_min + 1

    # Padding based on the person's size
    padding_x = int(person_width * PADDING_RATIO)
    padding_y = int(person_height * PADDING_RATIO)

    # Apply padding
    x_start = max(0, x_min - padding_x)
    x_end = min(image_width, x_max + padding_x + 1)

    y_start = max(0, y_min - padding_y)
    y_end = min(image_height, y_max + padding_y + 1)

    # Crop
    cropped_image = image[y_start:y_end, x_start:x_end]

    if cropped_image.size == 0:
        return None

    # Make the crop square without stretching
    crop_height, crop_width = cropped_image.shape[:2]
    canvas_size = max(crop_height, crop_width)

    top = (canvas_size - crop_height) // 2
    bottom = canvas_size - crop_height - top

    left = (canvas_size - crop_width) // 2
    right = canvas_size - crop_width - left

    centered_canvas = cv2.copyMakeBorder(
        cropped_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Resize while preserving proportions
    resized = cv2.resize(
        centered_canvas,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    return resized

def main():
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f'Input directory does not exist: {INPUT_DIR}'
        )

    if INPUT_DIR.resolve() == OUTPUT_DIR.resolve():
        raise ValueError(
            'Input and output directories must be different.'
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    processed_count = 0
    skipped_count = 0

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f'Skipped unreadable image: {image_path.name}')
            skipped_count += 1
            continue

        cleaned_image = crop_and_resize_person(image)

        if cleaned_image is None:
            print(
                f'Skipped image with no detectable foreground: '
                f'{image_path.name}'
            )
            skipped_count += 1
            continue

        output_path = OUTPUT_DIR / image_path.name

        if not cv2.imwrite(
            str(output_path),
            cleaned_image
        ):
            raise IOError(
                f'Failed to save cleaned image: {output_path}'
            )

        processed_count += 1

    print(
        f'Processed {processed_count} images; '
        f'skipped {skipped_count} images.'
    )

    print(f'New images saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
	main()
