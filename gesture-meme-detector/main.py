import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# ===================== НАСТРОЙКИ =====================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Мемы
memes = {
    "neutral": None,
    "hamster": cv2.imread("memes/hamster.jpg"),
    "girl": cv2.imread("memes/girl.jpg"),
    "cat": cv2.imread("memes/cat.jpg"),
    "sonic": cv2.imread("memes/sonic.jpg"),
    "shrug": cv2.imread("memes/shrug.jpg"),
    "dog": cv2.imread("memes/dog.jpg"),
}

gesture_buffer = deque(maxlen=10)
current_gesture = "neutral"

# ===================== МОДЕЛИ =====================
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Запущено!")

try:
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         FaceLandmarker.create_from_options(face_options) as face_landmarker:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detected = "neutral"

            # === ЛИЦО ===
            face_result = face_landmarker.detect(mp_image)
            if face_result.face_landmarks:
                lm = face_result.face_landmarks[0]

                if lm[14].y - lm[13].y > 0.045 or lm[15].y - lm[13].y > 0.04:
                    detected = "cat"

                nose_x = lm[1].x
                if abs(nose_x - lm[33].x) > 0.085 or abs(nose_x - lm[263].x) > 0.085:
                    detected = "dog"

            # === РУКИ + ТРЕКИНГ ===
            hand_result = hand_landmarker.detect(mp_image)
            num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0

            if hand_result.hand_landmarks:
                for hand_lm in hand_result.hand_landmarks:
                    lm = hand_lm
                    for i, landmark in enumerate(lm):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

                    fingers_up = 0
                    if lm[8].y < lm[6].y:   fingers_up += 1
                    if lm[12].y < lm[10].y: fingers_up += 1
                    if lm[16].y < lm[14].y: fingers_up += 1
                    if lm[20].y < lm[18].y: fingers_up += 1

                    if fingers_up == 1:
                        detected = "girl"
                    elif fingers_up == 2:
                        detected = "hamster"

            if num_hands >= 2 and face_result.face_landmarks and detected in ["neutral", "dog"]:
                lm_face = face_result.face_landmarks[0]
                face_top = lm_face[10].y
                face_center = lm_face[0].y

                if all(hand_lm[0].y < face_top + 0.18 for hand_lm in hand_result.hand_landmarks):
                    detected = "sonic"
                elif all(hand_lm[0].y > face_center - 0.05 for hand_lm in hand_result.hand_landmarks):
                    detected = "shrug"

            # Сглаживание
            gesture_buffer.append(detected)
            final_gesture = max(set(gesture_buffer), key=gesture_buffer.count)
            current_gesture = final_gesture

            # Отображение
            if current_gesture in memes and memes[current_gesture] is not None:
                meme = cv2.resize(memes[current_gesture], (w//2, h))
                combined = np.hstack((meme, frame))
            else:
                left = np.zeros((h, w//2, 3), dtype=np.uint8)
                cv2.putText(left, "neutral", (80, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (180, 180, 180), 4)
                combined = np.hstack((left, frame))

            cv2.imshow("Gesture Meme Detector", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
                break

except Exception as e:
    print("Ошибка:", e)
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена.")