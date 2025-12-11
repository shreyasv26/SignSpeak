# quick_test_model.py
import numpy as np
from tensorflow.keras.models import load_model

actions = ["hello","thank_you","yes","no","help","please","cat","eat","fine"]
model = load_model('action_model.h5')   # or action_model_final.h5 if that's what you have

# load first sequence of 'hello' (20 frames)
seq = [np.load(f"MP_Data/hello/0/{i}.npy") for i in range(20)]
pred = model.predict(np.expand_dims(np.array(seq), axis=0))[0]
print("Pred probs:", np.round(pred,3))
print("Predicted:", actions[pred.argmax()], "Confidence:", float(pred.max()))
