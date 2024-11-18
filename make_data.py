import cv2
import mediapipe as mp
import pandas as pd

# Initialize webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


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

            h, w, c = img.shape
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    return img


lm_list = []
label = "THREE"  # Đổi các cử chỉ khác
no_of_frames = 500

while len(lm_list) < no_of_frames:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        # Xử lý frame
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_RGB)

        # Nhận diện Hand landmarks:
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # Vẽ khung xương lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

        # Hiển thị frame, số frame đã count
        cv2.putText(frame, f'Frames: {len(lm_list)}/{no_of_frames}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (5, 5, 50), 2)

        cv2.imshow("Hand Gesture Recording", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Viết vào file csv
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()