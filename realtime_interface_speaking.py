# realtime_interface_speaking.py
# Realtime inference + speak only when label changes (non-blocking TTS)
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import threading
import pyttsx3

# ---------------- CONFIG ----------------
actions = ["hello","thank_you","I_Love_You","yes","no","help","please","cat","eat","fine"]
sequence_length = 20
model_path = "action_model.h5"
cap_index = 0

# smoothing / responsiveness
PROB_HISTORY = 2        # smaller -> faster reaction
LABEL_HISTORY = 4
CONF_THRESHOLD = 0.55
NO_CONF_TIMEOUT = 0.9   # seconds of low confidence before clearing label
# TTS cooldown (avoid rapid repeat)
SPEAK_COOLDOWN = 0.4    # seconds between identical spoken words
# debug
DEBUG = False
# ----------------------------------------

# Load model
model = load_model(model_path)

# Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# smoothing buffers
prob_hist = deque(maxlen=PROB_HISTORY)
label_hist = deque(maxlen=LABEL_HISTORY)

# ---------- Robust TTS helpers (replace your old TTS code) ----------
import threading, time
# remove global pyttsx3 init; we will init per thread to avoid engine state issues
_last_spoken_label = None
_last_spoken_time = 0.0
SPEAK_COOLDOWN = 1.0  # seconds between identical spoken words
DEBUG_TTS = True      # set True to print TTS debug info to console

def _speak_thread_new_engine(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()        # new engine per thread
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
    except Exception as e:
        print("TTS thread error:", e)

def maybe_speak(label):
    """
    Speak `label` in a background thread if cooldown allows.
    Uses a fresh engine per call to avoid engine re-use issues.
    """
    global _last_spoken_label, _last_spoken_time
    now = time.time()
    if label is None:
        return
    if label == _last_spoken_label and (now - _last_spoken_time) < SPEAK_COOLDOWN:
        if DEBUG_TTS:
            print(f"[TTS] Skipping repeat speak for '{label}' (cooldown).")
        return
    # update spoken info before starting thread so quick repeated triggers don't double-start
    _last_spoken_label = label
    _last_spoken_time = now
    spoken = label.replace('_', ' ')
    if DEBUG_TTS:
        print(f"[TTS] Speaking: '{spoken}'")
    th = threading.Thread(target=_speak_thread_new_engine, args=(spoken,), daemon=True)
    th.start()

# ---------- Normalization helper (same as training) ----------
def normalize_results_to_vector(results):
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

    # Normalize (mid-hip reference, scale by shoulder distance)
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
    lh_norm = ((lh2 - ref) / scale).flatten()
    rh_norm = ((rh2 - ref) / scale).flatten()
    return np.concatenate([pose_norm, lh_norm, rh_norm])

# ---------- BBox & draw helpers ----------
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

# ---------------- Main loop ----------------
def main():
    cap = cv2.VideoCapture(cap_index, cv2.CAP_DSHOW)
    seq = []
    last_label = None
    last_label_time = 0.0
    last_conf_time = 0.0
    previous_label = None
    prev = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
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
                # Move box ABOVE your head
                OFFSET = 70     # increase this value if you want it higher
                y1 = max(0, y1 - OFFSET)
                cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)

            # If no landmarks detected -> clear buffers to avoid sticky predictions
            if bbox is None:
                prob_hist.clear()
                label_hist.clear()
                last_label = None
                last_conf_time = 0.0

            # get normalized vector and build sequence
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
            # ----- REPLACE the prediction block with this -----
            # (inside the main loop where you already have seq and sequence_length)
            if len(seq) == sequence_length:
                t0 = time.time()
                res = model.predict(np.expand_dims(np.array(seq), axis=0), verbose=0)[0]   # current softmax
                pred_time = (time.time() - t0) * 1000.0

                # Append to prob history for display smoothing
                prob_hist.append(res)
                avg_res = np.mean(prob_hist, axis=0)

                # Smoothed confidence / index for display & label history
                sm_conf = float(np.max(avg_res))
                sm_idx = int(np.argmax(avg_res))

                # Current (instant) prediction using the most recent softmax
                cur_conf = float(np.max(res))
                cur_idx = int(np.argmax(res))
                cur_label = actions[cur_idx]
                sm_label = actions[sm_idx]

                # DEBUG prints (optional)
                if DEBUG:
                    print(f"[PRED] cur: ({cur_label},{cur_conf:.3f})  sm: ({sm_label},{sm_conf:.3f})")

                # Immediate speaking rule: if the *current* top label changed relative to last spoken
                # and current confidence is high enough, speak right away (subject to maybe_speak cooldown)
                if cur_conf >= CONF_THRESHOLD:
                    # speak the *current* label if it's different from last spoken - immediate
                    if cur_label != _last_spoken_label:
                        if DEBUG:
                            print(f"[SPEAK] immediate trigger for '{cur_label}' (conf {cur_conf:.3f})")
                        maybe_speak(cur_label)

                # Use smoothed label for display (reduces flicker)
                if sm_conf >= CONF_THRESHOLD:
                    # add to label history for majority vote display
                    label_hist.append(sm_idx)
                    voted = max(set(label_hist), key=label_hist.count)
                    last_label = actions[voted]
                    last_label_time = time.time()
                else:
                    # low smoothed confidence -> clear after timeout
                    if (time.time() - last_conf_time) > NO_CONF_TIMEOUT:
                        prob_hist.clear()
                        label_hist.clear()
                        last_label = None

                # update last_conf_time if smoothed conf is good
                if sm_conf >= CONF_THRESHOLD:
                    last_conf_time = time.time()

                # show latency
                cv2.putText(image, f"Latency: {int(pred_time)}ms", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)


            # display label if recent
            if last_label and (time.time() - last_label_time) < 1.2:
                lbl = last_label
                if bbox:
                    bx, by = x1, max(y1-28, 10)
                else:
                    bx, by = 10, 80
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(image, (bx-6, by-24), (bx+tw+6, by+6), (0,0,0), -1)
                cv2.putText(image, lbl, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            cv2.imshow("SignSpeak (speak on change)", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
