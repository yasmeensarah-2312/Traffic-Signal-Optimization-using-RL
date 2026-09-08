from ultralytics import YOLO
import cv2

# Load our trained ambulance model
model = YOLO("runs/detect/train/weights/best.pt")

# Input video
video_path = "data/ambulance_test2.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track objects
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.50
    )

    # Process detections
    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for i in range(len(boxes)):

            # Class ID
            class_id = int(boxes.cls[i])

            # Confidence
            confidence = float(boxes.conf[i])

            # Bounding box
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])

            # Track ID
            if boxes.id is not None:
                track_id = int(boxes.id[i])
            else:
                track_id = -1

            # Center coordinates
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Class name
            class_name = model.names[class_id]

            print(
                f"ID: {track_id} | "
                f"Class: {class_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Center: ({center_x}, {center_y})"
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display information
            label = f"{class_name} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display frame
    cv2.imshow("Ambulance Tracking", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()