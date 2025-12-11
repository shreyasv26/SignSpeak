# collect_single_word.py
# Interactive: pick one action at a time and record sequences
import cv2, os, numpy as np
import mediapipe as mp
from time import sleep

mp_holistic = mp.solutions.holistic

# ---------- CONFIG ----------
ACTIONS = ["hello","thank_you","I_Love_You","yes","no","help","please","cat","eat","fine"]
DATA_PATH = "MP_Data"
sequence_length = 20   # frames per sequence
no_sequences = 30      # recordings per action (set to 30)
# ----------------------------

def extract_keypoints(results):
    # pose (33 x (x,y,z,visibility))
    pose = np.zeros((33,4))
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
    # left hand (21 x (x,y,z))
    lh = np.zeros((21,3))
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    # right hand (21 x (x,y,z))
    rh = np.zeros((21,3))
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])  # length 258

# ensure folders
for a in ACTIONS:
    for s in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, a, str(s)), exist_ok=True)

# Interactive selection
print("Actions:")
for i,a in enumerate(ACTIONS):
    print(f"{i+1}. {a}")
choice = int(input("Enter number of action to record (1-{}): ".format(len(ACTIONS)))) - 1
action = ACTIONS[choice]
print(f"Selected: {action}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    print("Starting in 3 seconds. Get ready.")
    sleep(3)
    for seq in range(no_sequences):
        print(f"Recording {action} sequence {seq+1}/{no_sequences}")
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            keypoints = extract_keypoints(results)
            np.save(os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy"), keypoints)

            # show
            cv2.putText(image, f'{action} | Seq {seq+1}/{no_sequences} | Frame {frame_num+1}/{sequence_length}',
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow('Collecting - press q to stop', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        sleep(0.6)

cap.release()
cv2.destroyAllWindows()
print("Done collecting for", action)
