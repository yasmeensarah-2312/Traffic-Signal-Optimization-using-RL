from ultralytics import YOLO
import cv2
import os


# ==================================================
# SETTINGS
# ==================================================

MODEL_PATH = "runs/detect/ambulance_v1/weights/best.pt"
VIDEO_PATH = "data/ambulance_test2.mp4"

# YOLO confidence
CONFIDENCE = 0.50

# A detection must remain present for this many
# consecutive frames before we accept it.
REQUIRED_FRAMES = 10

# Minimum bounding-box area.
# Very small detections are usually unreliable.
MIN_BOX_AREA = 2500


# ==================================================
# LOAD MODEL
# ==================================================

print("Loading V1 ambulance model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")
print("Classes:", model.names)


# ==================================================
# OPEN VIDEO
# ==================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    exit()


# ==================================================
# TRACKING STATE
# ==================================================

candidate_id = None
candidate_frames = 0

ambulance_confirmed = False


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE,
        tracker="bytetrack.yaml",
        verbose=False
    )


    output = frame.copy()

    detected_candidate = False
    current_id = None


    # ==================================================
    # PROCESS DETECTIONS
    # ==================================================

    boxes = results[0].boxes


    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            confidence = float(box.conf[0])


            # ------------------------------------------
            # Bounding box
            # ------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                (x1, y1, x2, y2)
            )


            # ------------------------------------------
            # Box area
            # ------------------------------------------

            box_width = x2 - x1
            box_height = y2 - y1

            box_area = (
                box_width *
                box_height
            )


            # ------------------------------------------
            # Tracking ID
            # ------------------------------------------

            if box.id is not None:

                track_id = int(box.id[0])

            else:

                track_id = -1


            # ------------------------------------------
            # Candidate filter
            # ------------------------------------------

            if box_area < MIN_BOX_AREA:

                continue


            # ------------------------------------------
            # Candidate found
            # ------------------------------------------

            detected_candidate = True
            current_id = track_id


            # ------------------------------------------
            # Consecutive-frame confirmation
            # ------------------------------------------

            if current_id == candidate_id:

                candidate_frames += 1

            else:

                candidate_id = current_id
                candidate_frames = 1


            # ------------------------------------------
            # Confirm ambulance
            # ------------------------------------------

            if candidate_frames >= REQUIRED_FRAMES:

                ambulance_confirmed = True


            # ------------------------------------------
            # Draw detection
            # ------------------------------------------

            if ambulance_confirmed:

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


                label = (
                    f"AMBULANCE "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )


                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


                print(
                    f"AMBULANCE | "
                    f"ID:{track_id} | "
                    f"Confidence:{confidence:.2f} | "
                    f"Frames:{candidate_frames}"
                )


            break


    # ==================================================
    # RESET WHEN DETECTION DISAPPEARS
    # ==================================================

    if not detected_candidate:

        candidate_id = None
        candidate_frames = 0

        ambulance_confirmed = False


    # ==================================================
    # STATUS
    # ==================================================

    if ambulance_confirmed:

        status = "EMERGENCY: AMBULANCE DETECTED"

    else:

        status = "NORMAL: NO CONFIRMED AMBULANCE"


    cv2.putText(
        output,
        status,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ==================================================
    # DISPLAY
    # ==================================================

    cv2.imshow(
        "V1 Ambulance Detection - Filtered",
        output
    )


    # ==================================================
    # QUIT
    # ==================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==================================================
# CLEANUP
# ==================================================

cap.release()
cv2.destroyAllWindows()

print()
print("Detection completed.")