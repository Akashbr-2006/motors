import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def live_object_detection():
    # Start the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access the webcam")
        return

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

    
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

 
        results = model(rgb_frame)


        detections = results.xyxy[0].numpy()


        for det in detections:
            x_min, y_min, x_max, y_max, conf, cls = det
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            label = f"{results.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Live Object Detection", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


live_object_detection()
