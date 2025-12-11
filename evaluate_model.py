# evaluate_model.py
import os, numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

actions = ["hello","thank_you","I_Love_You","yes","no","help","please","cat","eat","fine"]
model = load_model('action_model.h5')

# load data (normalized same as training)
def normalize_frame(vec):
    pose = vec[:33*4].reshape(33,4)
    lh = vec[33*4:33*4+21*3].reshape(21,3)
    rh = vec[33*4+21*3:].reshape(21,3)
    try:
        ref = (pose[23,:3] + pose[24,:3]) / 2.0
    except:
        ref = np.array([0.5,0.5,0.0])
    shoulder_dist = np.linalg.norm(pose[11,:3] - pose[12,:3])
    scale = shoulder_dist if shoulder_dist > 1e-6 else 1.0
    pose_xyz = (pose[:,:3] - ref) / scale
    pose_vis = pose[:,3].reshape(33,1)
    pose_norm = np.concatenate([pose_xyz, pose_vis], axis=1).flatten()
    lh_norm = ((lh - ref) / scale).flatten()
    rh_norm = ((rh - ref) / scale).flatten()
    return np.concatenate([pose_norm, lh_norm, rh_norm])

X = []
y = []
for idx, a in enumerate(actions):
    ap = os.path.join('MP_Data', a)
    if not os.path.exists(ap):
        continue
    seqs = sorted([d for d in os.listdir(ap) if d.isdigit()], key=lambda x:int(x))
    for s in seqs:
        seq = [normalize_frame(np.load(os.path.join(ap,s,f"{i}.npy"))) for i in range(20)]
        X.append(seq)
        y.append(idx)

X = np.array(X)
y = np.array(y)

preds = model.predict(X)
y_pred = preds.argmax(axis=1)

print("Accuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n")
print(classification_report(y, y_pred, target_names=actions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y, y_pred))
