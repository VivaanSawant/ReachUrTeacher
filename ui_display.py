import cv2
from question_manager import face_images, face_positions

def display_question_queue():
    """Display faces in order of hand raising."""
    if face_images:
        max_faces_per_row = 6
        face_size = 100
        rows = []
        row = []

        for hand_id, (num, face_img) in face_images.items():
            face_resized = cv2.resize(face_img, (face_size, face_size))
            label = face_resized.copy()
            cv2.putText(label, f"#{num}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            combined = cv2.vconcat([label, face_resized])
            row.append(combined)

            if len(row) >= max_faces_per_row:
                rows.append(cv2.hconcat(row))
                row = []

        if row:
            rows.append(cv2.hconcat(row))

        face_display = cv2.vconcat(rows)
        cv2.imshow("Raised Hand Order", face_display)