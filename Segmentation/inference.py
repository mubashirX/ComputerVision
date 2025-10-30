from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/segment/train_yolov11_seg/weights/best.pt')

# Perform inference on the video
results = model.predict(source='input.mp4', save=True, show=True)
