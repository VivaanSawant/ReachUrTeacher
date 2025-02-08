import cv2
from hand_tracking import detect_hands, is_hand_above_face, is_open_palm
from face_detection import detect_faces
from question_manager import add_question, clear_all_questions, persistent_faces
from ui_display import display_question_queue

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_detected = detect_faces(frame, rgb_frame)
    hands_detected = detect_hands(frame, rgb_frame)

    for hand_landmarks in hands_detected:
        for (fx, fy, fw, fh) in faces_detected:
            face_top_y_normalized = fy / frame.shape[0]
            if is_open_palm(hand_landmarks) and is_hand_above_face(hand_landmarks, face_top_y_normalized):
                if (fx, fy, fw, fh) not in persistent_faces:
                    add_question((fx, fy, fw, fh), frame)

    display_question_queue()

    cv2.imshow("Face-Linked Hand Raising Order", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        clear_all_questions()

cap.release()
cv2.destroyAllWindows()