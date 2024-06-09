import cv2
import numpy as np
import os

# Load YOLOv4 model and classes
drive_path = r'/home/kartik/Downloads'  # Specify your local drive path here
weights_path = os.path.join(drive_path, 'yolov4.weights')
config_path = os.path.join(drive_path, 'yolov4.cfg')
names_path = os.path.join(drive_path, 'coco.names')
net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, "r") as f:
    classes = f.read().splitlines()

# Function to perform object detection on video stream
def detect_objects_from_video(input_source):
    if input_source == 'webcam':
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

    else:
        # Open video file
        cap = cv2.VideoCapture(input_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess input frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set input to the network
        net.setInput(blob)

        # Perform forward pass
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Process detections
        conf_threshold = 0.5
        nms_threshold = 0.4
        boxes = []
        confidences = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Perform non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Draw bounding boxes and labels
        for i in indices:
            i = int(i)
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the result
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Perform object detection on video file or webcam
    detect_objects_from_video('video1.mp4')  # Change to 'video.mp4' for video file input
