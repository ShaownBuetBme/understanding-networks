import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r'D:\shaown\Total project\3D\yolov8\yolov8_Checkerboard_human_best.pt')

# Read input image
image_path = r'D:\shaown\Total project\3D\anthrovision\front\53_f.jpg'
frame = cv2.imread(image_path)

# Check if image loaded
if frame is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Run inference
results = model(frame)[0]

# Draw bounding boxes and labels
for box in results.boxes:
    # Extract box coords and class id
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0].item()
    cls_id = int(box.cls[0].item())
    label = model.names[cls_id]

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put label and confidence
    text = f"{label} {conf:.2f}"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Convert BGR to RGB for matplotlib
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(frame_rgb)
plt.axis('off')
plt.title("YOLOv8 Detection")
plt.show()

# Save output image
output_path = r'D:\shaown\Total project\3D\yolov8\detection_output.jpg'
cv2.imwrite(output_path, frame)
print(f"Output saved to {output_path}")
