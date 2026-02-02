import supervision as sv

def create_tracker(
    fps: int,
    track_thresh: float = 0.25,
    track_buffer: int = 30,
    match_thresh: float = 0.8
) -> sv.ByteTrack:
    """
    Create and configure a ByteTrack tracker.

    Args:
        fps (int): Video frames per second.
        track_thresh (float): Confidence threshold for starting a track.
        track_buffer (int): Number of frames to keep lost tracks.
        match_thresh (float): IoU matching threshold.

    Returns:
        sv.ByteTrack: Configured ByteTrack tracker instance.
    """
    return sv.ByteTrack(
        frame_rate=fps,
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh
    )
