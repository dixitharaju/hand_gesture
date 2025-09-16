import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
import time

# Load model
gestures = ['volume_up', 'volume_down', 'mouse_move', 'mouse_right_click']
model = load_model("gesture_model.h5")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

screen_w, screen_h = pyautogui.size()
last_click_time = 0
click_delay = 1.0  # seconds

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

    # Flip frame for natural movement
    frame = cv2.flip(frame, 1)

    landmarks, hand_lms = get_landmarks(frame)
    if landmarks is not None:
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # ✅ Use flattened landmarks as input
        pred = model.predict(landmarks.reshape(1, -1), verbose=0)
        gesture_idx = np.argmax(pred)
        gesture_name = gestures[gesture_idx]

        cv2.putText(frame, gesture_name, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Perform actions
        if gesture_name == 'volume_up':
            pyautogui.press('volumeup')
        elif gesture_name == 'volume_down':
            pyautogui.press('volumedown')
        elif gesture_name == 'mouse_move':
            x = int(hand_lms.landmark[8].x * screen_w)
            y = int(hand_lms.landmark[8].y * screen_h)
            pyautogui.moveTo(x, y, duration=0.05)
        elif gesture_name == 'mouse_right_click':
            current_time = time.time()
            if current_time - last_click_time > click_delay:
                pyautogui.rightClick()
                last_click_time = current_time

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
