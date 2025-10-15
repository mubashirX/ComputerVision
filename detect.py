# ============================
# 📌 YOLO Video Detection Script
# ============================

from ultralytics import YOLO  # Import YOLO from ultralytics

# 1. 🔸 Load your trained model
model = YOLO("yolov8s.pt")  # Replace with your weights file path

# 2. 🔸 Run detection on your video
results = model.predict(
    source="input.mp4",   # Path to your input video
    conf=0.5,             # Confidence threshold (0 to 1)
    save=True,            # Save output video with bounding boxes
    device=0              # 0 for GPU, 'cpu' if no GPU
)

# 3. 📁 The output will be saved in the 'runs/detect' folder automatically
print("✅ Detection completed! Check the 'runs/detect' folder for results.")
