from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

for result in results:

    print("\nDetected objects:")

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = result.names[class_id]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        print(f"Object: {class_name}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"Center: ({center_x:.1f}, {center_y:.1f})")
        print("----------------------")