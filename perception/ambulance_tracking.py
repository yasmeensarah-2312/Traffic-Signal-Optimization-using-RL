from ultralytics import YOLO
import cv2
import sys
import os
import numpy as np

# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from simulation.signal_controller import get_signal_decision


# --------------------------------------------------
# Load trained ambulance model
# --------------------------------------------------

model = YOLO(
    "runs/detect/ambulance_v1/weights/best.pt"
)


# --------------------------------------------------
# Input video
# --------------------------------------------------

video_path = "data/ambulance_test2.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()


# --------------------------------------------------
# Emergency confirmation settings
# --------------------------------------------------

REQUIRED_FRAMES = 5

detection_count = 0

confirmed_lane = None

last_track_id = None


# --------------------------------------------------
# Traffic signal display
# --------------------------------------------------

def create_signal_display(signal_decision, confirmed):

    image = np.zeros(
        (450, 900, 3),
        dtype=np.uint8
    )

    cv2.putText(
        image,
        "EMERGENCY TRAFFIC SIGNAL CONTROL",
        (180, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    lanes = [
        ("LANE 1", "GREEN_LANE_1"),
        ("LANE 2", "GREEN_LANE_2"),
        ("LANE 3", "GREEN_LANE_3")
    ]

    positions = [150, 450, 750]

    for i, (lane_name, lane_signal) in enumerate(lanes):

        x = positions[i]

        cv2.putText(
            image,
            lane_name,
            (x - 55, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if confirmed and signal_decision == lane_signal:
            red_color = (0, 0, 0)
            green_color = (0, 255, 0)
        else:
            red_color = (0, 0, 255)
            green_color = (0, 0, 0)

        cv2.circle(
            image,
            (x, 180),
            40,
            red_color,
            -1
        )

        cv2.circle(
            image,
            (x, 320),
            40,
            green_color,
            -1
        )

    if confirmed:

        status = "AMBULANCE CONFIRMED"

    else:

        status = (
            f"VERIFYING AMBULANCE "
            f"({detection_count}/{REQUIRED_FRAMES})"
        )

    cv2.putText(
        image,
        status,
        (260, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return image


# --------------------------------------------------
# Process video
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # --------------------------------------------------
    # Lane boundaries
    # --------------------------------------------------

    lane1 = width // 3
    lane2 = (width // 3) * 2

    # --------------------------------------------------
    # YOLO tracking
    # --------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        conf=0.30,
        tracker="bytetrack.yaml",
        verbose=False
    )

    output = frame.copy()

    # --------------------------------------------------
    # Draw lane boundaries
    # --------------------------------------------------

    cv2.line(
        output,
        (lane1, 0),
        (lane1, height),
        (255, 255, 255),
        2
    )

    cv2.line(
        output,
        (lane2, 0),
        (lane2, height),
        (255, 255, 255),
        2
    )

    # --------------------------------------------------
    # Detection found in current frame
    # --------------------------------------------------

    current_detection = False

    current_lane = None

    current_id = None

    # --------------------------------------------------
    # Process detections
    # --------------------------------------------------

    if results[0].boxes is not None:

        for box in results[0].boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                (x1, y1, x2, y2)
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if box.id is not None:

                track_id = int(box.id[0])

            else:

                track_id = -1

            # --------------------------------------------------
            # Determine lane
            # --------------------------------------------------

            if center_x < lane1:

                lane = 1

            elif center_x < lane2:

                lane = 2

            else:

                lane = 3

            # --------------------------------------------------
            # Record detection
            # --------------------------------------------------

            current_detection = True
            current_lane = lane
            current_id = track_id

            # --------------------------------------------------
            # Draw bounding box
            # --------------------------------------------------

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Center point

            cv2.circle(
                output,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Label

            label = (
                f"ID:{track_id} "
                f"Ambulance "
                f"{confidence:.2f} "
                f"Lane:{lane}"
            )

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            break

    # --------------------------------------------------
    # Confirmation logic
    # --------------------------------------------------

    if current_detection:

        # Same tracking ID
        if current_id == last_track_id:

            detection_count += 1

        else:

            detection_count = 1

            confirmed_lane = None

        last_track_id = current_id

        # Confirm after required frames
        if detection_count >= REQUIRED_FRAMES:

            confirmed_lane = current_lane

    else:

        # No detection
        detection_count = 0
        confirmed_lane = None
        last_track_id = None

    # --------------------------------------------------
    # Signal decision
    # --------------------------------------------------

    if confirmed_lane is not None:

        signal_decision = get_signal_decision(
            confirmed_lane
        )

        confirmed = True

    else:

        signal_decision = "NORMAL_OPERATION"

        confirmed = False

    # --------------------------------------------------
    # Display system status
    # --------------------------------------------------

    cv2.putText(
        output,
        f"Signal: {signal_decision}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Confirmation: {detection_count}/{REQUIRED_FRAMES}",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------
    # Traffic signal window
    # --------------------------------------------------

    signal_window = create_signal_display(
        signal_decision,
        confirmed
    )

    # --------------------------------------------------
    # Show windows
    # --------------------------------------------------

    cv2.imshow(
        "Ambulance Detection",
        output
    )

    cv2.imshow(
        "Traffic Signal Simulation",
        signal_window
    )

    # --------------------------------------------------
    # Quit
    # --------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Emergency traffic system completed.")