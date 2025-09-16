import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

gesture_name = "mouse_right_click"  # change for each gesture
save_dir = f"landmark_dataset/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)
cap = cv2.VideoCapture(0)
count = 0

def get_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        landmarks = []
        for point in lm.landmark:
            landmarks.extend([point.x, point.y])
        return np.array(landmarks), lm
    return None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, hand_lms = get_landmarks(frame)
    if landmarks is not None:
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Press 'c' to capture {gesture_name}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Capture Hand Gestures", frame)

    key = cv2.waitKey(1)
    if key == ord('c') and landmarks is not None:
        np.save(f"{save_dir}/{count}.npy", landmarks)
        print(f"Saved {gesture_name}/{count}.npy")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
