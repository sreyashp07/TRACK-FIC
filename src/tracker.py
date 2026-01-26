import supervision as sv

def create_tracker(fps: int):
    return sv.ByteTrack(frame_rate=fps)
