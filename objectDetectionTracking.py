from ultralytics import YOLO
import cv2

print("Detection has started. Press 'q' to quit.")

# Load the YOLO model
model = YOLO("yolov8l.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize tracker
tracker = cv2.TrackerCSRT_create()

# Variables for tracking
is_tracking = False
bbox = None
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if not is_tracking:
        # Run YOLO detection
        results = model(frame)

        # Get the bounding box and detected classes from YOLO detection
        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]  # Get class name
                confidence = conf.item()  # Get confidence score

                # Print detected object information
                print(f"Detected: {label} with confidence: {confidence:.2f}")

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Make sure bbox has the form (x, y, width, height)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker.init(frame, bbox)  # Initialize the tracker
                is_tracking = True
                break  # Start tracking after detecting the first object

    else:
        # Periodic re-detection (every 30 frames, for example)
        frame_count += 1
        if frame_count % 30 == 0:
            results = model(frame)
            for r in results:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    tracker.init(frame, bbox)  # Reinitialize tracker with new bbox
                    is_tracking = True

        # Update tracker
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Object Detection and Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
