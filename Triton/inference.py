import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput

# ---------------- CONFIG ----------------
TRITON_URL = "localhost:8001"       # Triton server URL
MODEL_NAME = "person_detection"     # Your deployed model
INPUT_NAME = "images"               # Input tensor name in config.pbtxt
OUTPUT_NAME = "output0"             # Output tensor name in config.pbtxt

IMAGE_PATH = "images.jpeg"       # Path to your test image
MODEL_INPUT_H, MODEL_INPUT_W = 384, 640  # Must match your model input




# ---------------- CONNECT ----------------
client = grpcclient.InferenceServerClient(url=TRITON_URL)
outputs = [InferRequestedOutput(OUTPUT_NAME)]




# ---------------- LOAD IMAGE ----------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found!")


print("Original image shape:", img.shape)


# Resize to model input
inp_img = cv2.resize(img, (MODEL_INPUT_W, MODEL_INPUT_H))
print("Resized image shape:", inp_img.shape)

# Triton expects batch dimension
inp = np.expand_dims(inp_img, axis=0).astype(np.uint8)
print("Input tensor shape (with batch):", inp.shape)





# ---------------- PREPARE TRITON INPUT ----------------
input_tensor = InferInput(INPUT_NAME, inp.shape, "UINT8")
input_tensor.set_data_from_numpy(inp)


print("\nSending data to Triton...")
print("Input name:", INPUT_NAME)
print("Input shape:", inp.shape)
print("Input dtype:", inp.dtype)





# ---------------- RUN INFERENCE ----------------
response = client.infer(model_name=MODEL_NAME, inputs=[input_tensor], outputs=outputs)



# ---------------- FETCH OUTPUT ----------------
output0 = response.as_numpy(OUTPUT_NAME)


print("OUTPUT EXPLANATION")
print("Shape:", output0.shape)
print("Meaning: (batch_size=1, num_detections=300, values_per_detection=6)")
print("\nEach detection contains 6 values:")
print("  [x1, y1, x2, y2, confidence, class_id]")
print("="*60)


# Get detections from batch
detections = output0[0]  # Shape: (300, 6)


print("\nProcessing detections...")
print("Total predictions:", len(detections))


# Filter by confidence threshold
CONFIDENCE_THRESHOLD = 0.70
valid_detections = detections[detections[:, 4] > CONFIDENCE_THRESHOLD]

print("Detections above threshold (0.70):", len(valid_detections))
print("")



# Display each valid detection
for i, det in enumerate(valid_detections):
    x1, y1, x2, y2, conf, cls_id = det
    
    print("Detection", i+1)
    print("  Class ID:", int(cls_id))
    print("  Confidence:", round(float(conf), 4))
    print("  Bbox (resized image):", int(x1), int(y1), int(x2), int(y2))
    print("")



# Draw on image
img_with_boxes = img.copy()
scale_x = img.shape[1] / MODEL_INPUT_W
scale_y = img.shape[0] / MODEL_INPUT_H

for det in valid_detections:
    x1, y1, x2, y2, conf, cls_id = det
    
    # Scale back to original image size
    x1_orig = int(x1 * scale_x)
    y1_orig = int(y1 * scale_y)
    x2_orig = int(x2 * scale_x)
    y2_orig = int(y2 * scale_y)
    
    # Draw rectangle
    cv2.rectangle(img_with_boxes, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
    
    # Draw label
    label = "C" + str(int(cls_id)) + " " + str(round(float(conf), 2))
    cv2.putText(img_with_boxes, label, (x1_orig, y1_orig-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite("output_with_boxes.jpg", img_with_boxes)
print("Saved: output_with_boxes.jpg")