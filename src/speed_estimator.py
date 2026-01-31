def estimate_speed(
    y_history: list,
    fps: float,
    pixels_per_meter: float,
    window: int
) -> float | None:
    """
    Estimates speed (km/h) based on vertical pixel movement history.

    Args:
        y_history (list): List of y-coordinate positions over time.
        fps (float): Frames per second of the video.
        pixels_per_meter (float): Conversion factor from pixels to meters.
        window (int): Number of frames used for estimation.

    Returns:
        float | None: Estimated speed in km/h, or None if insufficient data.
    """

    # Not enough data to compute speed
    if len(y_history) < window:
        return None

    # Calculate pixel displacement
    start_y = y_history[0]
    end_y = y_history[-1]
    pixel_distance = abs(end_y - start_y)

    # Time taken for the window
    time_seconds = window / fps

    # Convert pixels to meters
    meters_travelled = pixel_distance / pixels_per_meter

    # Speed in km/h
    speed_kmh = (meters_travelled / time_seconds) * 3.6

    return speed_kmh
