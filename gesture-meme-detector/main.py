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

# Загрузка мемов с проверкой
memes = {}
meme_names = ["hamster", "girl", "cat", "sonic", "shrug", "dog"]

for name in meme_names:
    path = BASE_DIR / "memes" / f"{name}.jpg"
    img = cv2.imread(str(path))
    if img is not None:
        memes[name] = img
        print(f"✓ Загружен: {name}.jpg")
    else:
        print(f"⚠ Не найден: {name}.jpg")
        memes[name] = None

gesture_buffer = deque(maxlen=12)
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

print("✅ Программа запущена!")

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
                if abs(lm[1].x - 0.5) > 0.09:
                    detected = "dog"

            # === РУКИ ===
            hand_result = hand_landmarker.detect(mp_image)
            if hand_result.hand_landmarks:
                for hand_lm in hand_result.hand_landmarks:
                    lm = hand_lm
                    
                    # Трекинг рук
                    for i, landmark in enumerate(lm):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    fingers_up = sum(1 for i in [8,12,16,20] if lm[i].y < lm[i-2].y)

                    if fingers_up == 1:
                        detected = "girl"
                    elif fingers_up == 2:
                        detected = "hamster"

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

            # === ОТОБРАЖЕНИЕ МЕМА ===
            meme_img = memes.get(current_gesture)
            if meme_img is not None:
                meme = cv2.resize(meme_img, (w//2, h))
            else:
                meme = np.zeros((h, w//2, 3), dtype=np.uint8)
                cv2.putText(meme, current_gesture.upper(), (60, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 255), 3)

            combined = np.hstack((meme, frame))

            cv2.imshow("Gesture Meme Detector", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print("Ошибка:", e)
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена.")