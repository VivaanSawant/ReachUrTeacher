import cv2
from question_manager import persistent_people, assigned_numbers

def display_question_queue():
    """Displays faces in order of hand raising."""
    if persistent_people:
        max_faces_per_row = 6
        face_size = 100
        rows = []
        row = []

        for person_id, data in persistent_people.items():
            face_img = data['crop']
            num = assigned_numbers[person_id]

            face_resized = cv2.resize(face_img, (face_size, face_size))
            label = face_resized.copy()
            cv2.putText(label, f"#{num}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            row.append(cv2.vconcat([label, face_resized]))

            if len(row) >= max_faces_per_row:
                rows.append(cv2.hconcat(row))
                row = []

        if rows:
            cv2.imshow("Raised Hand Order", cv2.vconcat(rows))
