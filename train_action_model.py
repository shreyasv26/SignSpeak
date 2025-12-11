# train_action_model.py
# Run: conda activate action
#      python train_action_model.py

# train_action_model.py
import os, numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# ---------- CONFIG ----------
DATA_PATH = "MP_Data"
actions = ["hello","thank_you","I_Love_You","yes","no","help","please","cat","eat","fine"]
sequence_length = 20
# ----------------------------

# normalization helpers (expects vector shape (258,))
def normalize_frame(vec):
    # split
    pose = vec[:33*4].reshape(33,4)   # x,y,z,vis
    lh = vec[33*4:33*4 + 21*3].reshape(21,3)
    rh = vec[33*4 + 21*3:].reshape(21,3)

    # reference: mid-hip (left_hip idx 23, right_hip idx 24)
    try:
        ref = (pose[23,:3] + pose[24,:3]) / 2.0
    except Exception:
        ref = np.array([0.5, 0.5, 0.0])
    # scale: torso width = distance between shoulders (11 & 12)
    shoulder_dist = np.linalg.norm(pose[11,:3] - pose[12,:3])
    scale = shoulder_dist if shoulder_dist > 1e-6 else 1.0

    # normalize pose xyz, keep visibility as is
    pose_xyz = (pose[:,:3] - ref) / scale
    pose_vis = pose[:,3].reshape(33,1)
    pose_norm = np.concatenate([pose_xyz, pose_vis], axis=1).flatten()

    # normalize hands relative to same ref and scale
    lh_norm = ((lh - ref) / scale).flatten()
    rh_norm = ((rh - ref) / scale).flatten()

    return np.concatenate([pose_norm, lh_norm, rh_norm])

def load_data(data_path, actions, seq_len):
    X, y = [], []
    for idx, action in enumerate(actions):
        ap = os.path.join(data_path, action)
        if not os.path.exists(ap):
            print("Warning, missing", ap)
            continue
        seq_dirs = sorted([d for d in os.listdir(ap) if d.isdigit()], key=lambda x: int(x))
        for s in seq_dirs:
            frames = []
            for f in range(seq_len):
                p = os.path.join(ap, s, f"{f}.npy")
                if os.path.exists(p):
                    raw = np.load(p)
                else:
                    raw = np.zeros(33*4 + 21*3 + 21*3)
                frames.append(normalize_frame(raw))
            X.append(frames)
            y.append(idx)
    return np.array(X), np.array(y)

print("Loading data...")
X, y = load_data(DATA_PATH, actions, sequence_length)
print("X shape:", X.shape, "y shape:", y.shape)

if X.size == 0:
    raise SystemExit("No data found. Check MP_Data folder.")

y_cat = tf.keras.utils.to_categorical(y, num_classes=len(actions))

# compute class weights (helps classes that have fewer samples)
y_integers = y
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, shuffle=True)
print("Train/test shapes:", X_train.shape, X_test.shape)

n_features = X.shape[2]

# model
model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, n_features)),
    BatchNormalization(),
    LSTM(128, return_sequences=False, activation='tanh'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(len(actions), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# callbacks
checkpoint = ModelCheckpoint('action_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

# train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=80, batch_size=16, callbacks=[checkpoint, early], class_weight=class_weights)

model.save('action_model_final.h5')
print("Done. Models saved: action_model.h5, action_model_final.h5")
