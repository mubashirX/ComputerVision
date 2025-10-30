from ultralytics import YOLO
import os

# ==== CONFIG ====
DATA_YAML = "/home/mubashir/Desktop/self2/Segmentation/data/data.yaml"  # your dataset yaml path
MODEL_PATH = "yolo11n-seg.pt"          # will auto-download if missing
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0  # 0 for GPU, 'cpu' for CPU

# ==== TRAINING ====
if __name__ == "__main__":
    assert os.path.exists(DATA_YAML), f"❌ data.yaml not found at {DATA_YAML}"
    print(f"ℹ️ Using model: {MODEL_PATH} (will auto-download if missing)")
    print("🚀 Starting YOLOv11 segmentation training...")

    # Load model (auto-downloads if not found)
    model = YOLO(MODEL_PATH)

    # Train
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs/segment",
        name="train_yolov11_seg",
        exist_ok=True
    )

    print("\n✅ Training completed!")
    print("📁 Results saved to: runs/segment/train_yolov11_seg/")
