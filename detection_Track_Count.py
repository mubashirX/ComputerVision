"""
===========================================================
🎯 YOLO Object Detection + ByteTrack Tracking + Line Counting
-----------------------------------------------------------
✅ Runs YOLO detection internally
✅ Uses ByteTrack to assign IDs
✅ Draws bounding boxes + IDs
✅ Adds IN/OUT Line Counting using Supervision
✅ Saves tracked + annotated video
===========================================================
"""

# 📦 Libraries
import cv2
import torch
from ultralytics import YOLO
import supervision as sv  # 👈 NEW

# -----------------------------------------------------------
# 1. 🧠 Load YOLO Model
# -----------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8s.pt")

try:
    model.to(device)
except Exception:
    pass

# -----------------------------------------------------------
# 2. 🧭 Initialize Tracker
# -----------------------------------------------------------
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
# 4. 💾 Output Writer
# -----------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("detection_Track_count.mp4", fourcc, fps, (width, height))

# -----------------------------------------------------------
# 5. 🪄 Define Line Zone (only once)
# -----------------------------------------------------------
# Example: horizontal line in the middle of the frame
# start at left edge (x=0) and end at right edge (x=width)
start_point = sv.Point(0, height // 2)
end_point = sv.Point(width, height // 2)

line_zone = sv.LineZone(start=start_point, end=end_point)
line_annotator = sv.LineZoneAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

# Optional: box annotator for bounding boxes
box_annotator = sv.BoxAnnotator()

# -----------------------------------------------------------
# 6. 🧠 Detection + Tracking + Line Loop
# -----------------------------------------------------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video processing completed.")
        break

    frame_count += 1

    # --- YOLO Detection + Tracking ---
    results = model.track(
        frame,
        conf=0.5,
        tracker=tracker_config,
        persist=True,
        verbose=False,
        device=device
    )

    # --- Convert results to Supervision detections ---
    if results[0].boxes.id is not None:  # ensure IDs exist
        detections = sv.Detections.from_ultralytics(results[0])

        # --- Trigger line zone counting ---
        line_zone.trigger(detections)

        # --- Draw boxes ---
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        # --- Draw line + IN/OUT counts ---
        line_annotator.annotate(
            frame=annotated_frame,
            line_counter=line_zone
        )
    else:
        annotated_frame = frame.copy()

    # --- Write frame to output (no GUI display) ---
    out.write(annotated_frame)

    # Progress output (prints on one line)
    print(f"Processing frame {frame_count}/{total_frames}", end='\r')

# -----------------------------------------------------------
# 7. 🧹 Cleanup
# -----------------------------------------------------------
cap.release()
out.release()

print("🎉 Tracked + line-counting video saved to: detection_Track_count.mp4")
