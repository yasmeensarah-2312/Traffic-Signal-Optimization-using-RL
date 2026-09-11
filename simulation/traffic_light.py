import cv2
import numpy as np


def draw_traffic_lights(signal_decision):

    # Create black background
    image = np.zeros((500, 900, 3), dtype=np.uint8)

    # Title
    cv2.putText(
        image,
        "AUTOMATIC TRAFFIC SIGNAL CONTROL",
        (170, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    lanes = [
        ("LANE 1", "GREEN_LANE_1"),
        ("LANE 2", "GREEN_LANE_2"),
        ("LANE 3", "GREEN_LANE_3")
    ]

    x_positions = [150, 450, 750]

    for i, (lane_name, lane_signal) in enumerate(lanes):

        x = x_positions[i]

        # Lane name
        cv2.putText(
            image,
            lane_name,
            (x - 60, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Determine signal state
        if signal_decision == lane_signal:

            # Emergency lane = GREEN
            red = (0, 0, 0)
            yellow = (0, 0, 0)
            green = (0, 255, 0)

        else:

            # Other lanes = RED
            red = (0, 0, 255)
            yellow = (0, 0, 0)
            green = (0, 0, 0)

        # Red light
        cv2.circle(
            image,
            (x, 180),
            40,
            red,
            -1
        )

        # Yellow light
        cv2.circle(
            image,
            (x, 270),
            40,
            yellow,
            -1
        )

        # Green light
        cv2.circle(
            image,
            (x, 360),
            40,
            green,
            -1
        )

    # Display current decision
    cv2.putText(
        image,
        f"DECISION: {signal_decision}",
        (250, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return image


if __name__ == "__main__":

    # Test Lane 2 priority
    signal = "GREEN_LANE_2"

    output = draw_traffic_lights(signal)

    cv2.imshow(
        "Traffic Signal Simulation",
        output
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()