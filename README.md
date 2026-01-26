# TRACK-FIC  
### Vision-Based Vehicle Speed Estimation from Monocular Traffic Video

TRACK-FIC is a computer vision system designed to estimate per-vehicle speed from a single fixed camera feed.  
The project combines object detection, multi-object tracking, perspective transformation, and temporal motion analysis to produce stable and realistic speed estimates without relying on radar or sensor data.

---

## Key Capabilities
- Real-time vehicle detection using YOLOv8
- Persistent vehicle identity tracking via ByteTrack
- Perspective correction to mitigate camera-induced distortion
- Sliding-window speed estimation with noise suppression
- Frame-accurate speed annotation on output video

---

## Technical Overview
Vehicles are detected in each frame and tracked across time.  
Bottom-center anchor points of bounding boxes are projected into a bird’s-eye view using a homography transformation.  
Speed is estimated by measuring longitudinal displacement over a fixed temporal window and converting pixel motion into real-world units.

This approach ensures robustness against:
- Perspective compression
- Sudden detection jitter
- Short-lived false positives

---

## Technology Stack
- **Python**
- **YOLOv8 (Ultralytics)**
- **Supervision**
- **OpenCV**
- **NumPy**
- **ByteTrack**
- **Jupyter Notebook**

---

## Notebook Demonstration
An interactive notebook illustrating the full pipeline and visual outputs is provided:

📓 `notebooks/track_fic.ipynb`

(Originally prototyped in Google Colab and adapted for local execution.)

---

## Running Locally

1. Create a virtual environment  
2. Install dependencies:
3. Place your input video inside:
4. Run the pipeline:
5. The processed video with speed annotations
will be saved.



---

## Notes
- Speed calibration depends on the `PIXELS_PER_METER` parameter and camera geometry.
- Perspective source and target points must be adjusted per camera setup for optimal accuracy.

---

## License
This project is intended for academic, research, and demonstration purposes.
