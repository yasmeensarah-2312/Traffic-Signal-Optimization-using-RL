from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

for result in results:
    print("Detected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        print(
            f"{class_name} - confidence: {confidence:.2f}"
        )

    result.show()