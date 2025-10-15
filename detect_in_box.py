"""
===========================================================
🎯 YOLO Object Detection with ROI (Polygon Zone)
-----------------------------------------------------------
✅ Runs YOLO on full frame
✅ Filters detections inside ROI only
✅ Draws ROI polygon on video
✅ Saves output video
===========================================================
"""

# 📦 Required libraries
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv  # ✨ for ROI handling

# -----------------------------------------------------------
# 1. ✅ Load YOLO Model
# -----------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8s.pt")
try:
    model.to(device)
except Exception:
    pass

# -----------------------------------------------------------
# 2. 🎥 Open Input Video
# -----------------------------------------------------------
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

print(f"Video: {video_path}")
print(f"Resolution: {width}x{height} | FPS: {fps} | Total frames: {total_frames}")

# -----------------------------------------------------------
# 3. 🧭 Define ROI Polygon (center of the frame)
# -----------------------------------------------------------
# For example, a rectangle in the center of the frame
polygon = np.array([
    [width // 4, height // 4],
    [3 * width // 4, height // 4],
    [3 * width // 4, 3 * height // 4],
    [width // 4, 3 * height // 4]
])

# Create polygon zone
polygon_zone = sv.PolygonZone(polygon=polygon)

# Create annotator to draw the polygon outline
polygon_annotator = sv.PolygonZoneAnnotator(
    zone=polygon_zone,
    # use supervision Color constant so .as_bgr() is available
    color=sv.Color.RED,
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

# -----------------------------------------------------------
# 4. 💾 Prepare Output Writer
# -----------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("detect_in_box.mp4", fourcc, fps, (width, height))

# -----------------------------------------------------------
# 5. 🧠 Detection + ROI Filtering Loop
# -----------------------------------------------------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video processing completed.")
        break

    frame_count += 1

    # --- Run YOLO Detection ---
    results = model.predict(
        frame,
        conf=0.5,
        verbose=False,
        device=device
    )

    # Convert to Supervision Detections
    detections = sv.Detections.from_ultralytics(results[0])

    # --- 🧭 ROI Filter ---
    # Keep only detections inside the polygon
    mask = polygon_zone.trigger(detections=detections)
    detections = detections[mask]

    # --- Draw detections and ROI ---
    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(
        scene=annotated_frame,
        detections=detections
    )
    polygon_annotator.annotate(annotated_frame)

    # --- Write frame ---
    out.write(annotated_frame)

    if total_frames > 0:
        print(f"Processed {frame_count}/{total_frames} frames", end='\r')

# -----------------------------------------------------------
# 6. 🧹 Cleanup
# -----------------------------------------------------------
cap.release()
out.release()

print(f"🎉 Saved ROI-filtered detection video to: detect_in_box.mp4")
