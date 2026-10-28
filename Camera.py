import os
import time
from pathlib import Path
import cv2
import mediapipe as mp


def take_photo():
    """
    Clean, lightweight webcam capture:
    - No intrusive overlays or stencils.
    - Smooth countdown when the user is standing in front of the camera.
    - Press 'q' to quit.
    """
    mp_pose = mp.solutions.pose
    camera = cv2.VideoCapture(0)

    stable_start_time = None
    capture_saved = False
    photo_path = None
    photo_timestamp = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while camera.isOpened():
            success, raw_frame = camera.read()
            if not success:
                break

            display_frame = raw_frame.copy()
            h, w = raw_frame.shape[:2]
            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Basic presence check
                head = landmarks[mp_pose.PoseLandmark.NOSE]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

                person_detected = (
                    head.visibility > 0.6
                    and (left_hip.visibility > 0.6 or right_hip.visibility > 0.6)
                    and (left_foot.visibility > 0.6 or right_foot.visibility > 0.6)
                )

                if person_detected:
                    if stable_start_time is None:
                        stable_start_time = time.time()
                    elapsed = time.time() - stable_start_time

                    if elapsed >= 3.0 and not capture_saved:
                        photos_folder = Path("Photos")
                        photos_folder.mkdir(parents=True, exist_ok=True)
                        photo_timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"bmi_capture_{photo_timestamp}.jpg"
                        photo_path = str(photos_folder / filename)
                        capture_saved = cv2.imwrite(photo_path, raw_frame)
                        break

                    seconds_left = max(0.0, 3.0 - elapsed)
                    status_text = f"Hold still... Capturing in {seconds_left:.1f}s"
                    status_color = (0, 255, 0)
                else:
                    stable_start_time = None
                    status_text = "Ensure full body (head to feet) is in view"
                    status_color = (0, 165, 255)
            else:
                stable_start_time = None
                status_text = "Step in front of camera"
                status_color = (0, 0, 255)

            # Minimalist status banner
            cv2.rectangle(display_frame, (15, 15), (w - 15, 65), (0, 0, 0), -1)
            cv2.putText(
                display_frame,
                status_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA
            )

            cv2.imshow("BMI Capture System", display_frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()

    if capture_saved and photo_path is not None:
        saved_image = cv2.imread(photo_path)
        return saved_image, photo_timestamp

    return None
