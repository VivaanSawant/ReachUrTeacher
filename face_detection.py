import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

def detect_faces(frame, rgb_frame):
    """Detect faces in the frame and return bounding boxes."""
    face_results = face_detection.process(rgb_frame)
    faces_detected = []

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (
                int(bboxC.xmin * frame.shape[1]),
                int(bboxC.ymin * frame.shape[0]),
                int(bboxC.width * frame.shape[1]),
                int(bboxC.height * frame.shape[0])
            )

            y = max(0, y - int(0.3 * h))  # Adjust for hats
            faces_detected.append((x, y, w, h))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return faces_detected