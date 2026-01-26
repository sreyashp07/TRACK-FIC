def estimate_speed(
    y_history,
    fps,
    pixels_per_meter,
    window
):
    if len(y_history) < window:
        return None

    pixel_distance = abs(y_history[-1] - y_history[0])
    time_seconds = window / fps

    meters = pixel_distance / pixels_per_meter
    speed_kmh = (meters / time_seconds) * 3.6

    return speed_kmh
