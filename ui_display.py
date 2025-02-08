import cv2
from question_manager import persistent_people, assigned_numbers

def display_question_queue():
    """Display faces in order of hand raising."""
    if persistent_people:
        max_faces_per_row = 6
        face_size = 100
        rows = []
        row = []

        # Iterate through persistent_people to get face crops and assigned numbers
        for person_id, data in persistent_people.items():
            face_img = data['crop']
            num = assigned_numbers[person_id]  # Get the assigned number for this person

            # Resize the face image and add a label
            face_resized = cv2.resize(face_img, (face_size, face_size))
            label = face_resized.copy()
            cv2.putText(label, f"#{num}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            combined = cv2.vconcat([label, face_resized])
            row.append(combined)

            # Start a new row if the current row is full
            if len(row) >= max_faces_per_row:
                rows.append(cv2.hconcat(row))
                row = []

        # Add the last row if it's not empty
        if row:
            rows.append(cv2.hconcat(row))

        # Combine all rows into a single display image
        if rows:
            face_display = cv2.vconcat(rows)
            cv2.imshow("Raised Hand Order", face_display)