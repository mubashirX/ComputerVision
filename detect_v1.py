"""
===========================================================
🎯 YOLO Frame-by-Frame Object Detection (Beginner Friendly)
-----------------------------------------------------------
✅ Reads a video frame by frame
✅ Runs YOLO detection on each frame
✅ Draws bounding boxes and labels
✅ Displays and saves the output video

Author: Mubashir & ChatGPT
===========================================================
"""

# 📦 Required libraries
import cv2
import torch
from ultralytics import YOLO

# -----------------------------------------------------------
# 1. ✅ Load YOLO Model
# -----------------------------------------------------------
# Choose GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load your trained model (replace with your path)
model = YOLO("yolov8s.pt")

# Try to move model to selected device (some versions handle this internally)
try:
    model.to(device)
except Exception:
    pass

# -----------------------------------------------------------
# 2. 🎥 Open Input Video
# -----------------------------------------------------------
video_path = "input.mp4"   # 👉 replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {video_path}")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

print(f"Video: {video_path}")
print(f"Resolution: {width}x{height} | FPS: {fps} | Total frames: {total_frames}")

# -----------------------------------------------------------
# 3. 💾 Prepare Output Video Writer
# -----------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "detect_v1.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------------------------------------
# 4. 🧠 Frame-by-Frame Detection Loop
# -----------------------------------------------------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing completed.")
        break

    frame_count += 1

    # --- Run YOLO Detection on the current frame ---
    results = model.predict(
        frame,
        conf=0.5,           # Confidence threshold
        verbose=False,
        device=device
    )

    # --- Draw bounding boxes and labels ---
    annotated_frame = results[0].plot()

    # GUI display disabled (headless-safe). Frames are only written to the output file.

    # --- Write annotated frame to output video ---
    out.write(annotated_frame)

    # Optional: Print progress
    if total_frames > 0:
        print(f"Processed {frame_count}/{total_frames} frames", end='\r')

# -----------------------------------------------------------
# 5. 🧹 Cleanup
# -----------------------------------------------------------
cap.release()
out.release()

print(f"🎉 Saved output video to: {output_path}")
