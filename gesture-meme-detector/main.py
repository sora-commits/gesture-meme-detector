import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ===================== ПУТИ =====================
BASE_DIR = Path(__file__).parent.absolute()

hand_model_path = BASE_DIR / "hand_landmarker.task"
face_model_path = BASE_DIR / "face_landmarker.task"

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
    base_options=BaseOptions(model_asset_path=str(hand_model_path)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(face_model_path)),
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
                # Язык → Кот
                if lm[14].y - lm[13].y > 0.045 or lm[15].y - lm[13].y > 0.04:
                    detected = "cat"
                # Смотр в сторону → Собака
                if abs(lm[1].x - 0.5) > 0.09:
                    detected = "dog"

            # === РУКИ ===
            hand_result = hand_landmarker.detect(mp_image)
            if hand_result.hand_landmarks:
                for hand_lm in hand_result.hand_landmarks:
                    lm = hand_lm
                    
                    # Трекинг рук (видимый)
                    for i, landmark in enumerate(lm):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    # Подсчёт поднятых пальцев
                    fingers_up = 0
                    if lm[8].y < lm[6].y:   fingers_up += 1
                    if lm[12].y < lm[10].y: fingers_up += 1
                    if lm[16].y < lm[14].y: fingers_up += 1
                    if lm[20].y < lm[18].y: fingers_up += 1

                    if fingers_up == 1:
                        detected = "girl"
                    elif fingers_up == 2:
                        detected = "hamster"

            # Две руки
            num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
            if num_hands >= 2:
                if all(hand[0].y < 0.4 for hand in hand_result.hand_landmarks):
                    detected = "sonic"
                elif all(hand[0].y > 0.55 for hand in hand_result.hand_landmarks):
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
                cv2.putText(left, current_gesture.upper(), (70, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 3)
                combined = np.hstack((left, frame))

            cv2.imshow("Gesture Meme Detector", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print("Ошибка:", e)
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена.")