import cv2
from question_manager import face_images, face_positions, clear_question

def display_question_queue():
    """Display faces in order of hand raising."""
    if face_images:
        max_faces_per_row = 4
        face_size = 120
        rows = []
        row = []

        for index, (num, face_img) in enumerate(face_images):
            face_resized = cv2.resize(face_img, (face_size, face_size))
            label = cv2.putText(
                face_resized.copy(), f"#{num}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            combined = cv2.vconcat([label, face_resized])
            row.append(combined)

            if len(row) >= max_faces_per_row:
                rows.append(cv2.hconcat(row))
                row = []

        if row:
            rows.append(cv2.hconcat(row))

        face_display = cv2.vconcat(rows)
        cv2.imshow("Raised Hand Order", face_display)
        cv2.setMouseCallback("Raised Hand Order", on_mouse_click)

def on_mouse_click(event, x, y, flags, param):
    """Handle mouse click events to remove a face from the queue."""
    if event == cv2.EVENT_LBUTTONDOWN:
        for index, (px, py, pw, ph) in enumerate(face_positions):
            if px < x < px + pw and py < y < py + ph:
                clear_question(index)
                return