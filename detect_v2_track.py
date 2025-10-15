
"""
===========================================================
🎯 YOLO Object Detection + ByteTrack Tracking
-----------------------------------------------------------
✅ Runs YOLO detection internally
✅ Uses ByteTrack to assign IDs to objects
✅ Draws bounding boxes + IDs
✅ Saves tracked video
===========================================================
"""

# 📦 Libraries
import cv2
import torch
from ultralytics import YOLO

# -----------------------------------------------------------
# 1. 🧠 Load YOLO Model
# -----------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load your trained model (replace with your own path if needed)
model = YOLO("yolov8s.pt")

# Try moving model to GPU if available
try:
    model.to(device)
except Exception:
    pass

# -----------------------------------------------------------
# 2. 🧭 Initialize Tracker (ByteTrack)
# -----------------------------------------------------------
# Ultralytics provides a ready-made bytetrack.yaml file internally
tracker_config = "bytetrack.yaml"

# -----------------------------------------------------------
# 3. 🎥 Open Video File
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
# 4. 💾 Prepare Output Writer
# -----------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("detect_v2_track.mp4", fourcc, fps, (width, height))

# -----------------------------------------------------------
# 5. 🧠 Detection + Tracking Loop
# -----------------------------------------------------------
frame_count = 0

# We don't need a separate tracker object.
# YOLO will handle detection + tracking inside track()
while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video processing completed.")
        break

    frame_count += 1

    # *********** 🧠 TRACKING STEP ***********
    # model.track() = detect + track with ByteTrack internally
    results = model.track(
        frame,
        conf=0.5,
        tracker=tracker_config,
        persist=True,       # Keeps track of IDs across frames
        verbose=False,
        device=device
    )

    # results[0] contains detection + track IDs
    annotated_frame = results[0].plot()  # Draw bounding boxes and IDs

    # *********** Save annotated frame (GUI disabled) ***********
    # Don't call cv2.imshow / cv2.waitKey to remain headless-safe.
    out.write(annotated_frame)

    print(f"Processing frame {frame_count}/{total_frames}", end='\r')

# -----------------------------------------------------------
# 6. 🧹 Cleanup
# -----------------------------------------------------------
cap.release()
out.release()

print(f"🎉 Tracked video saved to: detect_v2_track.mp4")
