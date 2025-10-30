import cv2
import numpy as np
import os

# ==== CONFIG ====
images_path = "/home/mubashir/Desktop/self2/Segmentation/data/valid/images"       # your images folder
labels_path = "/home/mubashir/Desktop/self2/Segmentation/data/valid/labels"       # your YOLO segmentation .txt folder
output_path = "visualized_labels"    # where to save annotated images
os.makedirs(output_path, exist_ok=True)

# Optional: class names (edit if you have classes.txt or data.yaml)
classes = ['fire', 'smoke']

# ==== PROCESS ALL FILES ====
for label_file in os.listdir(labels_path):
    if not label_file.endswith(".txt"):
        continue

    image_file = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(images_path, image_file)
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Image not found for {label_file}")
        continue

    h, w, _ = img.shape

    with open(os.path.join(labels_path, label_file)) as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            cls_id = int(data[0])
            points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= w  # scale x
            points[:, 1] *= h  # scale y
            points = points.astype(np.int32)

            # Fill polygon (semi-transparent)
            overlay = img.copy()
            cv2.fillPoly(overlay, [points], color=(0, 255, 0))
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

            # Outline polygon
            cv2.polylines(img, [points], isClosed=True, color=(0,255,0), thickness=2)

            # Put class name
            label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            cv2.putText(img, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Save output image
    output_file = os.path.join(output_path, image_file)
    cv2.imwrite(output_file, img)
    print(f"✅ Saved: {output_file}")

print("\n🎉 All visualized images saved to:", os.path.abspath(output_path))
