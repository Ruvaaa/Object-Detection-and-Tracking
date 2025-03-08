from ultralytics import YOLO
import cv2
import time

print("Detection has started. Press 'q' to quit.")

# User input: How long should the program run? (in hours)
run_time_hours = float(input("Enter run time in hours (e.g., 3 for 3 hours): "))
run_time_seconds = run_time_hours * 3600  # Convert to seconds

# Load the YOLO model
model = YOLO("yolov8l.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize tracker
tracker = cv2.TrackerCSRT_create()

# Variables for tracking
is_tracking = False
bbox = None
start_time = time.time()  # Record start time
last_detection_time = start_time  # Track last detection time

while cap.isOpened():
    # Check if the specified time has elapsed
    elapsed_time = time.time() - start_time
    if elapsed_time >= run_time_seconds:
        print("Time limit reached. Exiting program.")
        break

    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()

    if not is_tracking or (current_time - last_detection_time >= 30):  # Run YOLO every 30 seconds
        last_detection_time = current_time  # Update last detection time

        # Run YOLO detection
        results = model(frame)

        # Process detections
        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]  # Get class name
                confidence = conf.item()  # Get confidence score

                # Print detected object information
                print(f"Detected: {label} with confidence: {confidence:.2f}")

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Convert bbox format (x, y, width, height)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker = cv2.TrackerCSRT_create()  # Ensure fresh tracker
                tracker.init(frame, bbox)  # Initialize tracker
                is_tracking = True
                break  # Track only the first detected object

    # Update tracker if initialized
    if is_tracking:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frame
    cv2.imshow("YOLO Object Detection and Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
