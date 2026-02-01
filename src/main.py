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
    # ---------------- LOAD MODEL ----------------
    model = YOLO(MODEL_NAME)

    # ---------------- VIDEO SETUP ----------------
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO)

    tracker = create_tracker(video_info.fps)

    # ---------------- VIEW TRANSFORMATION ----------------
    SOURCE_POINTS = np.array([
        [400, 720],
        [880, 720],
        [1200, 200],
        [80, 200]
    ])

    TARGET_POINTS = np.array([
        [0, 720],
        [400, 720],
        [400, 0],
        [0, 0]
    ])

    transformer = ViewTransformer(SOURCE_POINTS, TARGET_POINTS)

    # ---------------- ANNOTATION SETTINGS ----------------
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

    # ---------------- SPEED ESTIMATION STATE ----------------
    # For each tracker_id, store recent Y-coordinates
    coordinates_history = defaultdict(
        lambda: deque(maxlen=int(video_info.fps))
    )

    speed_window = int(video_info.fps * WINDOW_SECONDS)

    # ---------------- VIDEO PROCESSING ----------------
    with sv.VideoSink(TARGET_VIDEO, video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            # ---------- YOLO DETECTION ----------
            result = model(
                frame,
                imgsz=MODEL_RESOLUTION,
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(result)

            # ---------- FILTERING ----------
            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
            detections = detections[detections.class_id != 0]   # remove persons
            detections = detections.with_nms(IOU_THRESHOLD)

            # ---------- TRACKING ----------
            detections = tracker.update_with_detections(detections)

            # ---------- COORDINATE TRANSFORM ----------
            anchor_points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )

            transformed_points = transformer.transform_points(anchor_points)

            labels = [""] * len(detections)

            # ---------- SPEED COMPUTATION ----------
            for i, (track_id, point) in enumerate(
                zip(detections.tracker_id, transformed_points)
            ):
                current_y = int(point[1])

                # Smooth sudden jumps in detection
                if coordinates_history[track_id]:
                    last_y = coordinates_history[track_id][-1]
                    current_y = last_y + np.clip(
                        current_y - last_y,
                        -MAX_PIXEL_JUMP,
                        MAX_PIXEL_JUMP
                    )

                coordinates_history[track_id].append(current_y)

                speed = estimate_speed(
                    list(coordinates_history[track_id])[-speed_window:],
                    video_info.fps,
                    PIXELS_PER_METER,
                    speed_window
                )

                if speed and SPEED_MIN <= speed <= SPEED_MAX:
                    labels[i] = f"{int(speed)} km/h"

            # ---------- DRAW ANNOTATIONS ----------
            frame = trace_annotator.annotate(frame, detections)
            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels)

            sink.write_frame(frame)

    print("Processing complete.")


if __name__ == "__main__":
    main()
