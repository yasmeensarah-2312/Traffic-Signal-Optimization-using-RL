import cv2
import csv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_path = "data/traffic_test.mp4"
output_path = "data/traffic_data.csv"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()


def get_density(count):
    if count <= 2:
        return "LOW"
    elif count <= 5:
        return "MEDIUM"
    else:
        return "HIGH"


# Create CSV file
csv_file = open(output_path, "w", newline="")

writer = csv.writer(csv_file)

writer.writerow([
    "Frame",
    "Lane1",
    "Lane2",
    "Lane3",
    "Lane4",
    "Lane5"
])

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    height, width = frame.shape[:2]

    lane_width = width // 5

    lane_counts = [0, 0, 0, 0, 0]

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, class_id in zip(boxes, class_ids):

            # Vehicle classes
            if class_id not in [2, 3, 5, 7]:
                continue

            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2

            lane = int(center_x // lane_width)

            if lane >= 5:
                lane = 4

            lane_counts[lane] += 1

    # Save data
    writer.writerow([
        frame_number,
        lane_counts[0],
        lane_counts[1],
        lane_counts[2],
        lane_counts[3],
        lane_counts[4]
    ])

    # Draw lane boundaries
    for i in range(1, 5):

        x = i * lane_width

        cv2.line(
            annotated_frame,
            (x, 0),
            (x, height),
            (255, 0, 0),
            2
        )

    # Display lane information
    for i in range(5):

        density = get_density(lane_counts[i])

        x_position = i * lane_width + 20

        cv2.putText(
            annotated_frame,
            f"L{i + 1}: {lane_counts[i]}",
            (x_position, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            density,
            (x_position, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "Traffic Density",
        annotated_frame
    )

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("Traffic data saved successfully!")
print("File:", output_path)