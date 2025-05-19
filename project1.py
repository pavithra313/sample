import cv2
import torch

# Load pre-trained YOLOv5 model (you can fine-tune on traffic sign dataset like GTSDB or custom one)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use yolov5s for speed; change to custom model if trained

# Class labels (customize this if using your own dataset)
# These are sample traffic-related labels. Replace with your trained labels if using a custom dataset.
traffic_labels = ['stop sign', 'speed limit', 'no entry', 'traffic light']

# Capture video from webcam or file
cap = cv2.VideoCapture(0)  # Use 'video.mp4' for pre-recorded input

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Convert to pandas dataframe
    df = results.pandas().xyxy[0]

    # Draw bounding boxes and labels
    for i in range(len(df)):
        label = df.iloc[i]['name']
        conf = df.iloc[i]['confidence']
        if label in traffic_labels and conf > 0.5:
            x1, y1, x2, y2 = int(df.iloc[i]['xmin']), int(df.iloc[i]['ymin']), int(df.iloc[i]['xmax']), int(df.iloc[i]['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow('Traffic Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
