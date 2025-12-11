# realtime_interface_fixed.py
# Realtime inference with robust anti-stuck logic:
# - only accept labels when avg_conf >= CONF_THRESHOLD
# - clear buffers when no landmarks detected
# - decay label after a short no-confidence timeout
# - small smoothing windows by default (fast & responsive)

import cv2, time, numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque

# ---------------- CONFIG ----------------
actions = ["hello","thank_you","I_Love_You","yes","no","help","please","cat","eat","fine"]
sequence_length = 20
model_path = "action_model.h5"
cap_index = 0

# smoothing / responsiveness
PROB_HISTORY = 3        # avg last N softmax vectors
LABEL_HISTORY = 5       # majority vote window
CONF_THRESHOLD = 0.45   # require avg confidence >= this to accept label
NO_CONF_TIMEOUT = 0.9   # seconds to wait with low confidence before clearing label
# debug
DEBUG = False
# ----------------------------------------

model = load_model(model_path)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

prob_hist = deque(maxlen=PROB_HISTORY)
label_hist = deque(maxlen=LABEL_HISTORY)

def normalize_results_to_vector(results):
    # produce raw vector (same layout as saved .npy) then normalize
    pose = np.zeros((33,4))
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
    lh = np.zeros((21,3))
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    rh = np.zeros((21,3))
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    raw = np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])
    # normalize as in training
    pose2 = raw[:33*4].reshape(33,4)
    lh2 = raw[33*4:33*4+21*3].reshape(21,3)
    rh2 = raw[33*4+21*3:].reshape(21,3)
    try:
        ref = (pose2[23,:3] + pose2[24,:3]) / 2.0
    except:
        ref = np.array([0.5,0.5,0.0])
    shoulder_dist = np.linalg.norm(pose2[11,:3] - pose2[12,:3])
    scale = shoulder_dist if shoulder_dist > 1e-6 else 1.0
    pose_xyz = (pose2[:,:3] - ref) / scale
    pose_vis = pose2[:,3].reshape(33,1)
    pose_norm = np.concatenate([pose_xyz, pose_vis], axis=1).flatten()
    lh_norm = ((lh2 - ref)/scale).flatten()
    rh_norm = ((rh2 - ref)/scale).flatten()
    return np.concatenate([pose_norm, lh_norm, rh_norm])

def compute_bbox(results, image_shape):
    h, w = image_shape[:2]
    xs, ys = [], []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            xs.append(lm.x); ys.append(lm.y)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            xs.append(lm.x); ys.append(lm.y)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            xs.append(lm.x); ys.append(lm.y)
    if not xs:
        return None
    min_x = max(int(min(xs) * w) - 10, 0)
    max_x = min(int(max(xs) * w) + 10, w - 1)
    min_y = max(int(min(ys) * h) - 10, 0)
    max_y = min(int(max(ys) * h) + 10, h - 1)
    return (min_x, min_y, max_x, max_y)

def draw_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def main():
    cap = cv2.VideoCapture(cap_index, cv2.CAP_DSHOW)
    seq = []
    last_label = None
    last_label_time = 0.0
    last_conf_time = 0.0  # last time we had avg_conf >= threshold
    prev = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_landmarks(image, results)
            bbox = compute_bbox(results, image.shape)
            if bbox:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

            # if no landmarks detected -> clear buffers immediately to prevent sticky label
            if bbox is None:
                if DEBUG: print("No landmarks -> clearing buffers")
                prob_hist.clear()
                label_hist.clear()
                last_label = None
                last_conf_time = 0.0
            # build sequence vector (normalized)
            vec = normalize_results_to_vector(results)
            seq.append(vec)
            if len(seq) > sequence_length:
                seq = seq[-sequence_length:]

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev) if now - prev > 0 else 0.0
            prev = now
            cv2.putText(image, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Predict when we have a full sequence
            if len(seq) == sequence_length:
                t0 = time.time()
                res = model.predict(np.expand_dims(np.array(seq), axis=0), verbose=0)[0]
                pred_time = (time.time() - t0) * 1000.0
                prob_hist.append(res)
                avg_res = np.mean(prob_hist, axis=0)
                conf = float(np.max(avg_res))
                idx = int(np.argmax(avg_res))
                if DEBUG:
                    print("conf:", round(conf,3), "top:", actions[idx], "avg_res[top]:", round(avg_res[idx],3))

                # If confidence is high enough, update last_conf_time and allow label_hist to be appended
                if conf >= CONF_THRESHOLD:
                    last_conf_time = time.time()
                    label_hist.append(idx)
                    # majority vote
                    label = max(set(label_hist), key=label_hist.count)
                    last_label = actions[label]
                    last_label_time = time.time()
                else:
                    # low confidence -> do NOT append to label_hist
                    # If we've had low confidence for longer than NO_CONF_TIMEOUT, clear label
                    if (time.time() - last_conf_time) > NO_CONF_TIMEOUT:
                        if DEBUG: print("Low confidence timeout -> clearing label and histories")
                        prob_hist.clear()
                        label_hist.clear()
                        last_label = None

                # draw latency
                cv2.putText(image, f"Latency: {int(pred_time)}ms", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # show label if present recently
            if last_label and (time.time() - last_label_time) < 1.2:
                lbl = last_label
                if bbox:
                    bx, by = x1, max(y1-28, 10)
                else:
                    bx, by = 10, 80
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(image, (bx-6, by-24), (bx+tw+6, by+6), (0,0,0), -1)
                cv2.putText(image, lbl, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            cv2.imshow("SignSpeak (fixed anti-stuck)", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
