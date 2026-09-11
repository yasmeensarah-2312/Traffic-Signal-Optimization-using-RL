def get_signal_decision(lane):
    """
    Decide which traffic signal should receive priority
    based on the ambulance lane.
    """

    if lane == 1:
        return "GREEN_LANE_1"

    elif lane == 2:
        return "GREEN_LANE_2"

    elif lane == 3:
        return "GREEN_LANE_3"

    else:
        return "NORMAL_OPERATION"


# Test
if __name__ == "__main__":

    for lane in [1, 2, 3]:

        decision = get_signal_decision(lane)

        print(
            f"Ambulance detected in Lane {lane} "
            f"-> Signal Decision: {decision}"
        )