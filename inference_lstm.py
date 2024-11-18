import threading
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Load model
model = tf.keras.models.load_model("hand_gesture_model.h5")

label = "Warmup..."
lm_list = []
n_time_steps = 10

def make_landmark_timestep(results):
    c_lm = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, label,(10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(img, label,(10, 30), font,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label,(10, 30),font,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    max_prob = np.max(results[0])
    predicted_class = np.argmax(results, axis=1)[0]
    CONFIDENCE_THRESHOLD = 0.7
    labels = {
        0: "ONE",
        1: "TWO",
        2: "THREE",
        3: "FOUR",
        4: "PALM",
        5: "STOP",
        6: "OK",
        7: "CALL",
        8: "LIKE",
        9: "DISLIKE",
        10: "FIST",
        11: "MUTE",
        12: "PEACE",
        13: "ROCK",
        14: "GUN",
        15: "MINI HEART"
    }
    if max_prob < CONFIDENCE_THRESHOLD:
        label = "UNKNOWN"
    else:
        label = labels.get(predicted_class, "UNKNOWN")
    print(f"Confidence: {max_prob:.2f}, Prediction: {label}")
    return label


# Warmup
i = 0
warmup_frames = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_RGB)
    i += 1

    if i > warmup_frames:
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            if len(lm) > 0:
                lm_list.append(lm)
                if len(lm_list) == n_time_steps:
                    # Predict gesture in separate thread
                    thread = threading.Thread(target=detect, args=(model, lm_list))
                    thread.start()
                    lm_list = []

            frame = draw_landmark_on_image(mpDraw, results, frame)

        frame = draw_class_on_image(label, frame)
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()