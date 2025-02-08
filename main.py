import cv2
from face_detection import detect_faces, detect_hands, is_hand_above_face, is_open_palm

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # Enhance face detection for darker complexions
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and hands using the improved methods
    faces_detected = detect_faces(frame, rgb_frame)
    hands_detected = detect_hands(frame, rgb_frame)

    # Loop through detected hands and check if they are above any detected faces
    for hand_landmarks in hands_detected:
        for (fx, fy, fw, fh) in faces_detected:
            face_top_y_normalized = fy / frame.shape[0]  # Normalize y for hand detection check

            # Ensure the hand is above the face and somewhat open
            if is_open_palm(hand_landmarks) and is_hand_above_face(hand_landmarks, face_top_y_normalized):
                # Draw a marker above the face indicating a hand was raised
                cv2.putText(frame, "Hand Raised", (fx, fy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Show the main camera feed with detected faces and hands
    cv2.imshow("Face-Linked Hand Raising Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the application
        break

cap.release()
cv2.destroyAllWindows()