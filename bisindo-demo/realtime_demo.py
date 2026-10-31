"""
Demo realtime BISINDO gesture detection (webcam) memakai model dari notebook KawanIsyarat.

Persiapan:
  pip install mediapipe opencv-python tensorflow numpy
  Taruh file ini satu folder dengan:
    - bisindo_gesture_siformer.tflite
    - bisindo_labels.json
    - hand_landmarker.task   (download di notebook cell 2)
    - pose_landmarker.task   (download di notebook cell 2)

Jalankan: python realtime_demo.py   (tekan 'q' untuk keluar)
"""
import json
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SEQ_LEN = 30          # harus sama dengan training
FEAT_DIM = 108        # harus sama dengan training
BUFFER_FRAMES = 45    # jendela waktu ~1.5 detik @30fps; ubah sesuai durasi video training
CONF_THRESHOLD = 0.80 # di bawah ini dianggap "tidak ada gerakan"
SMOOTH_N = 5          # voting dari N prediksi terakhir

# ---------- Load model & label ----------
with open("bisindo_labels.json", encoding="utf-8") as f:
    LABELS = {int(k): v for k, v in json.load(f).items()}

interpreter = tf.lite.Interpreter(model_path="bisindo_gesture_siformer.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

# ---------- MediaPipe ----------
hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)
pose_landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker.task"),
        min_pose_detection_confidence=0.5,
    )
)


# ---------- Normalisasi: SALIN PERSIS dari notebook (cell 2) ----------
def normalize_siformer_style(pose_result, hand_result):
    body_features = np.zeros(24)
    l_hand_features = np.zeros(42)
    r_hand_features = np.zeros(42)

    if pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        neck_x = (lms[11].x + lms[12].x) / 2
        neck_y = (lms[11].y + lms[12].y) / 2
        head_metric = np.sqrt((lms[11].x - lms[12].x) ** 2 + (lms[11].y - lms[12].y) ** 2)
        if head_metric == 0:
            head_metric = np.sqrt((neck_x - lms[0].x) ** 2 + (neck_y - lms[0].y) ** 2)

        if head_metric > 0:
            start_x = neck_x - 3 * head_metric
            start_y = lms[2].y + head_metric
            end_x = neck_x + 3 * head_metric
            end_y = start_y - 6 * head_metric

            body_pts = [
                (lms[0].x, lms[0].y), (neck_x, neck_y), (lms[5].x, lms[5].y),
                (lms[2].x, lms[2].y), (lms[8].x, lms[8].y), (lms[7].x, lms[7].y),
                (lms[12].x, lms[12].y), (lms[11].x, lms[11].y), (lms[14].x, lms[14].y),
                (lms[13].x, lms[13].y), (lms[16].x, lms[16].y), (lms[15].x, lms[15].y),
            ]
            norm_body = []
            for x, y in body_pts:
                nx = (x - start_x) / (end_x - start_x) if end_x != start_x else 0
                ny = (y - end_y) / (start_y - end_y) if start_y != end_y else 0
                norm_body.extend([nx, ny])
            body_features = np.array(norm_body)

    def normalize_hand(hand_lms):
        xs = [lm.x for lm in hand_lms]
        ys = [lm.y for lm in hand_lms]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if width > height:
            delta_x = 0.1 * width
            delta_y = delta_x + ((width - height) / 2)
        else:
            delta_y = 0.1 * height
            delta_x = delta_y + ((height - width) / 2)
        start_x, end_x = min(xs) - delta_x, max(xs) + delta_x
        start_y, end_y = min(ys) - delta_y, max(ys) + delta_y
        norm_hand = []
        for lm in hand_lms:
            nx = (lm.x - start_x) / (end_x - start_x) if end_x != start_x else 0
            ny = (lm.y - start_y) / (end_y - start_y) if end_y != start_y else 0
            norm_hand.extend([nx, ny])
        return np.array(norm_hand)

    if hand_result.hand_landmarks:
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            if i < len(hand_result.handedness):
                label = hand_result.handedness[i][0].category_name
                if label == "Left":
                    l_hand_features = normalize_hand(hand_lms)
                else:
                    r_hand_features = normalize_hand(hand_lms)

    return np.concatenate([body_features, l_hand_features, r_hand_features])


def predict(buffer):
    """buffer: list fitur per frame -> (label, confidence)"""
    idx = np.linspace(0, len(buffer) - 1, SEQ_LEN, dtype=int)  # sama seperti training
    seq = np.array([buffer[i] for i in idx], dtype=np.float32)[None, ...]  # (1,30,108)
    interpreter.set_tensor(inp["index"], seq)
    interpreter.invoke()
    probs = interpreter.get_tensor(out["index"])[0]
    k = int(np.argmax(probs))
    return k, float(probs[k])


def main():
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=BUFFER_FRAMES)
    recent = deque(maxlen=SMOOTH_N)
    shown = "..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Model diberi frame ASLI (tidak di-flip) agar konsisten dengan data training.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        feats = normalize_siformer_style(
            pose_landmarker.detect(mp_image), hand_landmarker.detect(mp_image)
        )
        buffer.append(feats)

        conf = None
        if len(buffer) == BUFFER_FRAMES:
            k, conf = predict(list(buffer))
            recent.append(k if conf >= CONF_THRESHOLD else -1)
            vote = max(set(recent), key=recent.count)
            shown = LABELS[vote] if vote != -1 else "..."

        # Flip DULU (hanya untuk tampilan), baru tulis teks supaya tidak terbalik.
        display = cv2.flip(frame, 1)
        if conf is not None:
            cv2.putText(display, f"conf {conf:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display, shown, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("BISINDO realtime", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()