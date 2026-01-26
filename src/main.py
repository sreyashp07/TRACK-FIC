from ultralytics import YOLO
import supervision as sv
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque

from config import *
from transformer import ViewTransformer
from tracker import create_tracker
from speed_estimator import estimate_speed

SOURCE_VIDEO = "data/input/sample_input.mp4"
TARGET_VIDEO = "data/output/sample_output.mp4"

def main():
    model = YOLO(MODEL_NAME)

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
    frames = sv.get_video_frames_generator(SOURCE_VIDEO)

    tracker = create_tracker(video_info.fps)

    SOURCE = np.array([[400,720],[880,720],[1200,200],[80,200]])
    TARGET = np.array([[0,720],[400,720],[400,0],[0,0]])
    transformer = ViewTransformer(SOURCE, TARGET)

    width, height = video_info.resolution_wh
    thickness = max(1, int((width + height) / 1000))
    text_scale = max(0.4, (width + height) / 3000)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * TRACE_SECONDS),
        position=sv.Position.BOTTOM_CENTER
    )

    coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))
    window = int(video_info.fps * WINDOW_SECONDS)

    with sv.VideoSink(TARGET_VIDEO, video_info) as sink:
        for frame in tqdm(frames, total=video_info.total_frames):
            result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
            detections = detections[detections.class_id != 0]
            detections = detections.with_nms(IOU_THRESHOLD)
            detections = tracker.update_with_detections(detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = transformer.transform_points(points)

            labels = [""] * len(detections)

            for i, (tid, point) in enumerate(zip(detections.tracker_id, points)):
                y = int(point[1])

                if coordinates[tid]:
                    last = coordinates[tid][-1]
                    y = last + np.clip(y - last, -MAX_PIXEL_JUMP, MAX_PIXEL_JUMP)

                coordinates[tid].append(y)

                speed = estimate_speed(
                    list(coordinates[tid])[-window:],
                    video_info.fps,
                    PIXELS_PER_METER,
                    window
                )

                if speed and SPEED_MIN <= speed <= SPEED_MAX:
                    labels[i] = f"{int(speed)} km/h"

            frame = trace_annotator.annotate(frame, detections)
            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels)
            sink.write_frame(frame)

    print("Processing complete.")

if __name__ == "__main__":
    main()
